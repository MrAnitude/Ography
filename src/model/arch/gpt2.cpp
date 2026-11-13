#include "model/arch/gpt2.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "kernels/kernels.h"

namespace ography {

namespace {

// View of the first n rows of a max_tokens-row scratch buffer.
Tensor rows(const Tensor& t, int n) { return t.slice_rows(0, n); }

}  // namespace

Gpt2Model::Gpt2Model(ModelConfig config, Gpt2Weights weights, DeviceArena& scratch_arena,
                     cublasHandle_t cublas, int max_tokens)
    : config_(std::move(config)),
      weights_(std::move(weights)),
      cublas_(cublas),
      max_tokens_(max_tokens) {
  const int64_t T = max_tokens;
  const int64_t H = config_.hidden_size;
  residual_ = scratch_arena.allocate(DType::F32, {T, H});
  x_ = scratch_arena.allocate(DType::F16, {T, H});
  qkv_ = scratch_arena.allocate(DType::F16, {T, 3 * H});
  q_ = scratch_arena.allocate(DType::F16, {T, config_.num_heads, config_.head_dim});
  attn_ = scratch_arena.allocate(DType::F16, {T, config_.num_heads, config_.head_dim});
  proj_ = scratch_arena.allocate(DType::F16, {T, H});
  fc_ = scratch_arena.allocate(DType::F16, {T, config_.intermediate_size});
}

void Gpt2Model::forward(const ForwardInput& input, KVCache& kv_cache, Tensor& logits,
                        cudaStream_t stream) {
  const int T = input.n_tokens;
  if (T <= 0 || T > max_tokens_) {
    throw std::invalid_argument("Gpt2Model::forward: n_tokens " + std::to_string(T) +
                                " outside [1, " + std::to_string(max_tokens_) + "]");
  }
  if (input.start_pos + T > config_.max_position_embeddings || input.start_pos + T > kv_cache.capacity()) {
    throw std::out_of_range("Gpt2Model::forward: positions up to " +
                            std::to_string(input.start_pos + T) + " exceed the context limit");
  }
  if (input.token_ids.numel() != T || input.positions.numel() != T) {
    throw std::invalid_argument("Gpt2Model::forward: token_ids/positions must have n_tokens rows");
  }
  if (logits.dtype() != DType::F32 || logits.numel() != config_.vocab_size) {
    throw std::invalid_argument("Gpt2Model::forward: logits must be [vocab] f32");
  }

  const float eps = config_.norm_eps;
  const float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
  const Tensor& ids = input.token_ids;
  const Tensor& pos = input.positions;

  Tensor residual = rows(residual_, T);
  Tensor x = rows(x_, T);
  Tensor qkv = rows(qkv_, T);
  Tensor q = rows(q_, T);
  Tensor attn = rows(attn_, T);
  Tensor proj = rows(proj_, T);
  Tensor fc = rows(fc_, T);
  // Attention output viewed as [T, H] for the c_proj GEMM.
  Tensor attn_2d(attn.data(), DType::F16, {T, config_.hidden_size}, Device::CUDA);

  kernels::embedding_with_positions(weights_.wte, weights_.wpe, ids, pos, residual, stream);
  observe(0, residual, stream);
  kernels::layernorm(residual, weights_.layers[0].ln_1_weight, weights_.layers[0].ln_1_bias, eps,
                     x, stream);

  const int n_layers = static_cast<int>(weights_.layers.size());
  for (int i = 0; i < n_layers; ++i) {
    const Gpt2LayerWeights& L = weights_.layers[i];

    kernels::gemm(cublas_, x, L.wqkv.weight, qkv, stream);
    kernels::split_qkv(qkv, L.bqkv, config_.num_heads, config_.num_kv_heads, q, kv_cache.k(i),
                       kv_cache.v(i), pos, /*inv_freq=*/Tensor(), stream);
    if (input.is_decode) {
      kernels::attention_decode(q, kv_cache.k(i), kv_cache.v(i), pos, scale, attn, stream);
    } else {
      kernels::attention_prefill(q, kv_cache.k(i), kv_cache.v(i), pos, scale, attn, stream);
    }
    kernels::gemm(cublas_, attn_2d, L.wo.weight, proj, stream);
    kernels::add_layernorm(proj, L.bo, residual, L.ln_2_weight, L.ln_2_bias, eps, x, stream);

    kernels::gemm(cublas_, x, L.w_fc.weight, fc, stream);
    kernels::bias_gelu(fc, L.b_fc, stream);
    kernels::gemm(cublas_, fc, L.w_proj.weight, proj, stream);

    const bool last = i + 1 == n_layers;
    const Tensor& next_w = last ? weights_.ln_f_weight : weights_.layers[i + 1].ln_1_weight;
    const Tensor& next_b = last ? weights_.ln_f_bias : weights_.layers[i + 1].ln_1_bias;
    kernels::add_layernorm(proj, L.b_proj, residual, next_w, next_b, eps, x, stream);
    observe(i + 1, residual, stream);
  }
  observe(n_layers + 1, x, stream);

  // x now holds ln_f(residual). Only the last token's logits are needed.
  Tensor last = x.slice_rows(T - 1, 1);
  Tensor logits_2d(logits.data(), DType::F32, {1, config_.vocab_size}, Device::CUDA);
  kernels::gemm(cublas_, last, weights_.lm_head.weight, logits_2d, stream);
}

}  // namespace ography
