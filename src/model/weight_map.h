#pragma once

#include <vector>

#include "core/memory.h"
#include "model/config.h"
#include "model/safetensors.h"
#include "quant/quant_config.h"

namespace ography {

// Weights in the engine's own layout, resident on the GPU.
// Projections are fused where it saves kernel launches at batch size 1:
//   wqkv    = concat(q_proj, k_proj, v_proj)  along out_features
//   w_gate_up = concat(gate_proj, up_proj)    along out_features
struct LayerWeights {
  Tensor attn_norm;
  QuantizedWeight wqkv;
  Tensor bqkv;  // only when config.attention_bias
  QuantizedWeight wo;
  Tensor ffn_norm;
  QuantizedWeight w_gate_up;
  QuantizedWeight w_down;
};

struct ModelWeights {
  Tensor embed_tokens;
  std::vector<LayerWeights> layers;
  Tensor final_norm;
  QuantizedWeight lm_head;  // aliases embed_tokens when tie_word_embeddings
};

// HF tensor names -> ModelWeights. The name scheme is per architecture
// (e.g. "model.layers.{i}.self_attn.q_proj.weight" for Llama).
// Reads from the mmapped checkpoint, fuses projections, converts dtype if needed,
// and uploads into the arena.
ModelWeights map_weights(const SafetensorsModel& checkpoint, const ModelConfig& config,
                         DeviceArena& arena);

}  // namespace ography
