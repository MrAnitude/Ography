#include "model/config.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace ography {

namespace {

using nlohmann::json;

// HF configs use explicit nulls for "not set" (e.g. GPT-2's n_inner, Llama's rope_scaling),
// so treat null the same as a missing key.
template <typename T>
T get_or(const json& j, const char* key, T fallback) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  return it->get<T>();
}

template <typename T>
T require(const json& j, const char* key, const std::string& path) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) {
    throw std::runtime_error(path + ": missing required key \"" + key + "\"");
  }
  return it->get<T>();
}

// eos_token_id may be an int or a list of ints (Llama 3); keep the first.
int first_token_id(const json& j, const char* key) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return -1;
  if (it->is_array()) return it->empty() ? -1 : (*it)[0].get<int>();
  return it->get<int>();
}

DType dtype_from_torch(const std::string& s, const std::string& path) {
  if (s == "bfloat16") return DType::BF16;
  if (s == "float16") return DType::F16;
  if (s == "float32") return DType::F32;
  throw std::runtime_error(path + ": unsupported torch_dtype \"" + s + "\"");
}

void parse_gpt2(const json& j, const std::string& path, ModelConfig& c) {
  c.vocab_size = require<int>(j, "vocab_size", path);
  c.hidden_size = require<int>(j, "n_embd", path);
  c.num_layers = require<int>(j, "n_layer", path);
  c.num_heads = require<int>(j, "n_head", path);
  c.num_kv_heads = c.num_heads;
  c.head_dim = c.hidden_size / c.num_heads;
  c.intermediate_size = get_or<int>(j, "n_inner", 4 * c.hidden_size);
  c.max_position_embeddings = require<int>(j, "n_positions", path);
  c.norm_eps = get_or<float>(j, "layer_norm_epsilon", 1e-5f);
  c.hidden_act = get_or<std::string>(j, "activation_function", "gelu_new");
  c.tie_word_embeddings = get_or<bool>(j, "tie_word_embeddings", true);
  c.attention_bias = true;  // every GPT-2 linear layer has a bias
  // GPT-2 checkpoints predate torch_dtype and are stored in fp32.
  c.dtype = dtype_from_torch(get_or<std::string>(j, "torch_dtype", "float32"), path);
}

void parse_llama_like(const json& j, const std::string& path, ModelConfig& c) {
  c.vocab_size = require<int>(j, "vocab_size", path);
  c.hidden_size = require<int>(j, "hidden_size", path);
  c.intermediate_size = require<int>(j, "intermediate_size", path);
  c.num_layers = require<int>(j, "num_hidden_layers", path);
  c.num_heads = require<int>(j, "num_attention_heads", path);
  c.num_kv_heads = get_or<int>(j, "num_key_value_heads", c.num_heads);
  c.head_dim = get_or<int>(j, "head_dim", c.hidden_size / c.num_heads);
  c.max_position_embeddings = require<int>(j, "max_position_embeddings", path);
  c.norm_eps = get_or<float>(j, "rms_norm_eps", 1e-5f);
  c.hidden_act = get_or<std::string>(j, "hidden_act", "silu");
  c.rope_theta = get_or<float>(j, "rope_theta", 10000.0f);
  c.tie_word_embeddings = get_or<bool>(j, "tie_word_embeddings", false);
  // Qwen2 always has q/k/v biases but doesn't say so in its config.
  c.attention_bias = get_or<bool>(j, "attention_bias", c.model_type == "qwen2");
  c.dtype = dtype_from_torch(get_or<std::string>(j, "torch_dtype", "bfloat16"), path);

  auto rs = j.find("rope_scaling");
  if (rs != j.end() && rs->is_object()) {
    // Newer configs say "rope_type", older ones "type".
    c.rope_scaling.type = get_or<std::string>(*rs, "rope_type", get_or<std::string>(*rs, "type", ""));
    c.rope_scaling.factor = get_or<float>(*rs, "factor", 1.0f);
    c.rope_scaling.low_freq_factor = get_or<float>(*rs, "low_freq_factor", 1.0f);
    c.rope_scaling.high_freq_factor = get_or<float>(*rs, "high_freq_factor", 4.0f);
    c.rope_scaling.original_max_position_embeddings =
        get_or<int>(*rs, "original_max_position_embeddings", 0);
  }
}

}  // namespace

ModelConfig ModelConfig::from_file(const std::string& config_json_path) {
  std::ifstream in(config_json_path);
  if (!in) throw std::runtime_error("cannot open " + config_json_path);

  json j;
  try {
    j = json::parse(in);
  } catch (const json::exception& e) {
    throw std::runtime_error(config_json_path + ": " + e.what());
  }

  ModelConfig c;
  auto archs = j.find("architectures");
  if (archs != j.end() && archs->is_array() && !archs->empty()) {
    c.architecture = (*archs)[0].get<std::string>();
  }
  c.model_type = get_or<std::string>(j, "model_type", "");

  try {
    if (c.model_type == "gpt2") {
      parse_gpt2(j, config_json_path, c);
    } else {
      parse_llama_like(j, config_json_path, c);
    }
  } catch (const json::exception& e) {
    throw std::runtime_error(config_json_path + ": " + e.what());
  }

  c.bos_token_id = first_token_id(j, "bos_token_id");
  c.eos_token_id = first_token_id(j, "eos_token_id");

  if (c.num_heads <= 0 || c.num_kv_heads <= 0 || c.num_heads % c.num_kv_heads != 0) {
    throw std::runtime_error(config_json_path + ": invalid head counts");
  }
  c.quant = QuantConfig::from_hf_config(j);
  return c;
}

}  // namespace ography
