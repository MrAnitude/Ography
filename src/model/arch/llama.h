#pragma once

#include <cublas_v2.h>

#include "core/memory.h"
#include "model/model.h"
#include "model/weight_map.h"

namespace ography {

// Llama-family decoder (also covers Mistral and Qwen2, which differ only in config:
// GQA ratio, rope settings, q/k/v bias, tied embeddings).
//
// The residual stream is fp32; everything else is fp16. Per layer, where x is the
// RMSNorm'd residual coming in:
//
//   qkv  = x @ Wqkv^T                         gemm
//   q, K/V cache <- rope(qkv + bqkv)          split_qkv (bias only for Qwen2)
//   a    = attention(q, cache)                attention_prefill / attention_decode
//   o    = a @ Wo^T                           gemm
//   residual += o;  x = ffn_norm(residual)    add_rmsnorm
//   f    = silu(x @ Wgate^T) * (x @ Wup^T)    gemm (fused gate_up), swiglu
//   d    = f @ Wdown^T                        gemm
//   residual += d;  x = next attn_norm (or final norm)   add_rmsnorm
class LlamaModel final : public Model {
 public:
  // Scratch activations for up to max_tokens tokens per forward pass come from scratch_arena.
  LlamaModel(ModelConfig config, LlamaWeights weights, DeviceArena& scratch_arena,
             cublasHandle_t cublas, int max_tokens = 512);

  // Same contract as Gpt2Model::forward.
  void forward(const ForwardInput& input, KVCache& kv_cache, Tensor& logits,
               cudaStream_t stream) override;
  const ModelConfig& config() const override { return config_; }

 private:
  ModelConfig config_;
  LlamaWeights weights_;
  cublasHandle_t cublas_;
  int max_tokens_;
  Tensor inv_freq_;  // [head_dim / 2] f32, rope_scaling applied

  // Scratch, each sized for max_tokens_ rows.
  Tensor residual_;  // [T, H] f32
  Tensor x_;         // [T, H] f16, norm output feeding the next GEMM
  Tensor qkv_;       // [T, (nh + 2 nkv) * hd] f16
  Tensor q_;         // [T, nh, hd] f16
  Tensor attn_;      // [T, nh, hd] f16
  Tensor proj_;      // [T, H] f16, output of o_proj / down_proj
  Tensor gate_up_;   // [T, 2I] f16
  Tensor act_;       // [T, I] f16
};

}  // namespace ography
