#include "model/arch/llama.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "core/cuda_check.h"
#include "kernels/kernels.h"
#include "model/linear.h"
#include "model/rope.h"

namespace ography {

LlamaModel::LlamaModel(ModelConfig config, LlamaWeights weights, DeviceArena& scratch_arena,
                       cublasHandle_t cublas, int max_tokens)
    : config_(std::move(config)),
      weights_(std::move(weights)),
      cublas_(cublas),
      max_tokens_(max_tokens) {
  const int64_t T = max_tokens;
  const int64_t H = config_.hidden_size;
  const int64_t nh = config_.num_heads, nkv = config_.num_kv_heads, hd = config_.head_dim;
  const int64_t I = config_.intermediate_size;
  if (config_.hidden_act != "silu") {
    throw std::runtime_error("LlamaModel: hidden_act \"" + config_.hidden_act + "\" unsupported");
  }

  const std::vector<float> inv_freq = rope_inv_freq(config_);
  inv_freq_ = scratch_arena.allocate(DType::F32, {hd / 2});
  CUDA_CHECK(cudaMemcpy(inv_freq_.data(), inv_freq.data(), inv_freq_.nbytes(),
                        cudaMemcpyHostToDevice));

  residual_ = scratch_arena.allocate(DType::F32, {T, H});
  x_ = scratch_arena.allocate(DType::F16, {T, H});
  qkv_ = scratch_arena.allocate(DType::F16, {T, (nh + 2 * nkv) * hd});
  q_ = scratch_arena.allocate(DType::F16, {T, nh, hd});
  attn_ = scratch_arena.allocate(DType::F16, {T, nh, hd});
  proj_ = scratch_arena.allocate(DType::F16, {T, H});
  gate_up_ = scratch_arena.allocate(DType::F16, {T, 2 * I});
  act_ = scratch_arena.allocate(DType::F16, {T, I});
  if (config_.quant.method != QuantMethod::None) {
    // INT4 prefill dequantizes weight rows into this in tiles, each then multiplied by cuBLAS.
    // 8 MiB stays in the 32 MB L2, so the fp16 tile never goes to DRAM: measured on
    // Qwen2.5 7B AWQ, a 30-token prefill takes 69 ms with 8 MiB tiles vs 130 ms with 64 MiB.
    dequant_ = scratch_arena.allocate(DType::F16, {int64_t{4} << 20});
  }
}

void LlamaModel::forward(const ForwardInput& input, KVCache& kv_cache, Tensor& logits,
                         cudaStream_t stream) {
  const int T = input.n_tokens;
  if (T <= 0 || T > max_tokens_) {
    throw std::invalid_argument("LlamaModel::forward: n_tokens " + std::to_string(T) +
                                " outside [1, " + std::to_string(max_tokens_) + "]");
  }
  if (input.start_pos + T > kv_cache.capacity()) {
    throw std::out_of_range("LlamaModel::forward: positions up to " +
                            std::to_string(input.start_pos + T) + " exceed the KV cache");
  }
  if (input.token_ids.numel() != T || input.positions.numel() != T) {
    throw std::invalid_argument("LlamaModel::forward: token_ids/positions must have n_tokens rows");
  }
  if (logits.dtype() != DType::F32 || logits.numel() != config_.vocab_size) {
    throw std::invalid_argument("LlamaModel::forward: logits must be [vocab] f32");
  }

  const float eps = config_.norm_eps;
  const float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));
  const Tensor& pos = input.positions;
  auto rows = [T](const Tensor& t) { return t.slice_rows(0, T); };

  Tensor residual = rows(residual_), x = rows(x_), qkv = rows(qkv_), q = rows(q_);
  Tensor attn = rows(attn_), proj = rows(proj_), gate_up = rows(gate_up_), act = rows(act_);
  Tensor attn_2d(attn.data(), DType::F16,
                 {T, int64_t(config_.num_heads) * config_.head_dim}, Device::CUDA);

  kernels::embedding(weights_.embed_tokens, input.token_ids, residual, stream);
  observe(0, residual, stream);
  kernels::rmsnorm(residual, weights_.layers[0].attn_norm, eps, x, stream);

  const int n_layers = static_cast<int>(weights_.layers.size());
  for (int i = 0; i < n_layers; ++i) {
    const LlamaLayerWeights& L = weights_.layers[i];

    linear(cublas_, x, L.wqkv, qkv, dequant_, stream);
    kernels::split_qkv(qkv, L.bqkv, config_.num_heads, config_.num_kv_heads, q, kv_cache.k(i),
                       kv_cache.v(i), pos, inv_freq_, stream);
    if (input.is_decode) {
      kernels::attention_decode(q, kv_cache.k(i), kv_cache.v(i), pos, scale, attn, stream);
    } else {
      kernels::attention_prefill(q, kv_cache.k(i), kv_cache.v(i), pos, scale, attn, stream);
    }
    linear(cublas_, attn_2d, L.wo, proj, dequant_, stream);
    kernels::add_rmsnorm(proj, residual, L.ffn_norm, eps, x, stream);

    linear(cublas_, x, L.w_gate_up, gate_up, dequant_, stream);
    kernels::swiglu(gate_up, act, stream);
    linear(cublas_, act, L.w_down, proj, dequant_, stream);

    const Tensor& next_norm = i + 1 == n_layers ? weights_.final_norm : weights_.layers[i + 1].attn_norm;
    kernels::add_rmsnorm(proj, residual, next_norm, eps, x, stream);
    observe(i + 1, residual, stream);
  }
  observe(n_layers + 1, x, stream);

  // x now holds final_norm(residual). Only the last token's logits are needed.
  Tensor last = x.slice_rows(T - 1, 1);
  Tensor logits_2d(logits.data(), DType::F32, {1, config_.vocab_size}, Device::CUDA);
  kernels::gemm(cublas_, last, weights_.lm_head.weight, logits_2d, stream);
}

}  // namespace ography
