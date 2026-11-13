#pragma once

#include <string>

#include "core/dtype.h"
#include "quant/quant_config.h"

namespace ography {

struct RopeScaling {
  std::string type;  // "", "llama3", "linear", "yarn", ...
  float factor = 1.0f;
  float low_freq_factor = 1.0f;
  float high_freq_factor = 4.0f;
  int original_max_position_embeddings = 0;
};

// Everything the engine needs from a HF config.json. Field names follow Llama's config;
// other families' keys (GPT-2's n_embd, n_layer, ...) are mapped onto them in from_file().
struct ModelConfig {
  std::string architecture;  // e.g. "LlamaForCausalLM", selects the arch in registry.cpp
  std::string model_type;    // e.g. "llama", "gpt2", "qwen2"
  int vocab_size = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_layers = 0;
  int num_heads = 0;
  int num_kv_heads = 0;      // < num_heads => grouped-query attention
  int head_dim = 0;          // defaults to hidden_size / num_heads
  int max_position_embeddings = 0;
  float norm_eps = 1e-5f;    // RMSNorm eps (Llama) or LayerNorm eps (GPT-2)
  std::string hidden_act;    // "silu" (SwiGLU), "gelu_new" (GPT-2 tanh GELU), ...
  float rope_theta = 10000.0f;
  RopeScaling rope_scaling;
  bool tie_word_embeddings = false;
  bool attention_bias = false;  // true for Qwen2-style q/k/v biases
  int bos_token_id = -1;
  int eos_token_id = -1;
  DType dtype = DType::BF16;    // checkpoint dtype (torch_dtype); compute dtype is chosen at load
  QuantConfig quant;

  // Throws std::runtime_error if the file can't be read or a required key is missing.
  static ModelConfig from_file(const std::string& config_json_path);
};

}  // namespace ography
