#pragma once

#include <cublas_v2.h>

#include "core/memory.h"
#include "model/model.h"
#include "model/weight_map.h"

namespace ography {

// GPT-2 (pre-LayerNorm decoder, learned positions, GELU MLP, biases everywhere).
//
// The residual stream is fp32; everything else is fp16. Per layer, where x is the
// LayerNorm'd residual coming in:
//
//   qkv  = x @ Wqkv^T                     gemm
//   q, K/V cache <- qkv + bqkv            split_qkv
//   a    = attention(q, cache)            attention_prefill / attention_decode
//   o    = a @ Wo^T                       gemm
//   residual += o + bo;  x = ln_2(residual)              add_layernorm
//   f    = gelu(x @ Wfc^T + bfc)          gemm, bias_gelu
//   p    = f @ Wproj^T                    gemm
//   residual += p + bproj;  x = next ln_1 (or ln_f)      add_layernorm
//
// The first ln_1 runs on the embeddings; the last add_layernorm applies ln_f, after which only
// the last token goes through lm_head.
class Gpt2Model final : public Model {
 public:
  // Scratch activations for up to max_tokens tokens per forward pass are allocated from
  // scratch_arena. Longer prompts must be fed in chunks.
  Gpt2Model(ModelConfig config, Gpt2Weights weights, DeviceArena& scratch_arena,
            cublasHandle_t cublas, int max_tokens = 512);

  // Writes K/V for input.positions into kv_cache but does not advance it; the caller does
  // that once the pass is done. logits: [vocab] fp32, for the last input token.
  void forward(const ForwardInput& input, KVCache& kv_cache, Tensor& logits,
               cudaStream_t stream) override;
  const ModelConfig& config() const override { return config_; }

  int max_tokens() const { return max_tokens_; }

 private:
  ModelConfig config_;
  Gpt2Weights weights_;
  cublasHandle_t cublas_;
  int max_tokens_;

  // Scratch, each sized for max_tokens_ rows.
  Tensor residual_;  // [T, H] f32
  Tensor x_;         // [T, H] f16, LayerNorm output feeding the next GEMM
  Tensor qkv_;       // [T, 3H] f16
  Tensor q_;         // [T, n_heads, head_dim] f16
  Tensor attn_;      // [T, n_heads, head_dim] f16
  Tensor proj_;      // [T, H] f16, output of c_proj (attention and MLP)
  Tensor fc_;        // [T, I] f16
};

}  // namespace ography
