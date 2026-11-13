#pragma once

#include <vector>

#include "core/memory.h"
#include "model/config.h"
#include "model/safetensors.h"
#include "quant/quant_config.h"

namespace ography {

// Weights in the engine's own layout, resident on the GPU. One struct per architecture.
// All linear weights are stored [out_features, in_features] (see kernels.h), whatever the
// checkpoint layout, and converted to the compute dtype (fp16 for now) at load time.

// ---- Llama family ----
// Projections are fused where it saves kernel launches at batch size 1:
//   wqkv    = concat(q_proj, k_proj, v_proj)  along out_features
//   w_gate_up = concat(gate_proj, up_proj)    along out_features
struct LlamaLayerWeights {
  Tensor attn_norm;
  QuantizedWeight wqkv;
  Tensor bqkv;  // only when config.attention_bias
  QuantizedWeight wo;
  Tensor ffn_norm;
  QuantizedWeight w_gate_up;
  QuantizedWeight w_down;
};

struct LlamaWeights {
  Tensor embed_tokens;
  std::vector<LlamaLayerWeights> layers;
  Tensor final_norm;
  QuantizedWeight lm_head;  // aliases embed_tokens when tie_word_embeddings
};

// HF tensor names -> LlamaWeights. The name scheme differs slightly per family
// (e.g. "model.layers.{i}.self_attn.q_proj.weight" for Llama).
// Reads from the mmapped checkpoint, fuses projections, converts dtype if needed,
// and uploads into the arena.
LlamaWeights map_llama_weights(const SafetensorsModel& checkpoint, const ModelConfig& config,
                               DeviceArena& arena);

// ---- GPT-2 ----
// HF stores GPT-2 linears as Conv1D, i.e. [in, out]; they are transposed to [out, in] here.
// c_attn already holds q, k and v fused along out_features.
struct Gpt2LayerWeights {
  Tensor ln_1_weight, ln_1_bias;
  QuantizedWeight wqkv;  // [3*hidden, hidden]
  Tensor bqkv;           // [3*hidden]
  QuantizedWeight wo;    // [hidden, hidden]
  Tensor bo;
  Tensor ln_2_weight, ln_2_bias;
  QuantizedWeight w_fc;  // [intermediate, hidden]
  Tensor b_fc;
  QuantizedWeight w_proj;  // [hidden, intermediate]
  Tensor b_proj;
};

struct Gpt2Weights {
  Tensor wte;  // [vocab, hidden] token embeddings
  Tensor wpe;  // [max_positions, hidden] learned position embeddings
  std::vector<Gpt2LayerWeights> layers;
  Tensor ln_f_weight, ln_f_bias;
  QuantizedWeight lm_head;  // aliases wte (GPT-2 always ties them)
};

// Accepts both bare names ("h.0.attn.c_attn.weight") and the "transformer." prefix some
// GPT-2 checkpoints use. Skips the "h.{i}.attn.bias" causal-mask buffers.
Gpt2Weights map_gpt2_weights(const SafetensorsModel& checkpoint, const ModelConfig& config,
                             DeviceArena& arena, DType compute_dtype = DType::F16);

// Copies a CPU tensor (F32/F16/BF16) to a new arena tensor of dst_dtype, optionally
// transposing a 2-d tensor. Exposed for tests and other architectures' loaders.
Tensor upload_tensor(const Tensor& src, DType dst_dtype, bool transpose, DeviceArena& arena);

}  // namespace ography
