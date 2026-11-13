#pragma once

#include <cublas_v2.h>

#include "core/memory.h"
#include "model/model.h"
#include "model/weight_map.h"

namespace ography {

// Llama-family decoder (also covers Mistral and Qwen2, which differ only in config:
// GQA ratio, rope settings, q/k/v bias, tied embeddings).
//
// Per layer:  x -> add_rmsnorm -> wqkv -> rope -> kv_cache_write -> attention -> wo
//               -> add_rmsnorm -> w_gate_up -> swiglu -> w_down
class LlamaModel final : public Model {
 public:
  LlamaModel(ModelConfig config, LlamaWeights weights, DeviceArena& scratch_arena,
             cublasHandle_t cublas);

  void forward(const ForwardInput& input, KVCache& kv_cache, Tensor& logits,
               cudaStream_t stream) override;
  const ModelConfig& config() const override { return config_; }

 private:
  ModelConfig config_;
  LlamaWeights weights_;
  cublasHandle_t cublas_;
  Tensor inv_freq_;  // precomputed rope frequencies (with rope_scaling applied)
  // TODO: activation scratch buffers sized for max prefill chunk.
};

}  // namespace ography
