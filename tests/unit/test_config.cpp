#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "model/config.h"

using namespace ography;
namespace fs = std::filesystem;

namespace {

ModelConfig parse(const std::string& json_text) {
  const fs::path p = fs::path(::testing::TempDir()) / "ography_config_test.json";
  std::ofstream(p) << json_text;
  ModelConfig c = ModelConfig::from_file(p.string());
  fs::remove(p);
  return c;
}

}  // namespace

TEST(ModelConfig, Gpt2) {
  // Trimmed openai-community/gpt2 config.json.
  const auto c = parse(R"({
    "activation_function": "gelu_new", "architectures": ["GPT2LMHeadModel"],
    "bos_token_id": 50256, "eos_token_id": 50256, "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2", "n_ctx": 1024, "n_embd": 768, "n_head": 12, "n_inner": null,
    "n_layer": 12, "n_positions": 1024, "vocab_size": 50257 })");
  EXPECT_EQ(c.architecture, "GPT2LMHeadModel");
  EXPECT_EQ(c.model_type, "gpt2");
  EXPECT_EQ(c.vocab_size, 50257);
  EXPECT_EQ(c.hidden_size, 768);
  EXPECT_EQ(c.intermediate_size, 3072);  // n_inner null -> 4 * n_embd
  EXPECT_EQ(c.num_layers, 12);
  EXPECT_EQ(c.num_heads, 12);
  EXPECT_EQ(c.num_kv_heads, 12);
  EXPECT_EQ(c.head_dim, 64);
  EXPECT_EQ(c.max_position_embeddings, 1024);
  EXPECT_FLOAT_EQ(c.norm_eps, 1e-5f);
  EXPECT_EQ(c.hidden_act, "gelu_new");
  EXPECT_TRUE(c.tie_word_embeddings);
  EXPECT_TRUE(c.attention_bias);
  EXPECT_EQ(c.dtype, DType::F32);
  EXPECT_EQ(c.eos_token_id, 50256);
}

TEST(ModelConfig, LlamaWithGqaAndRopeScaling) {
  const auto c = parse(R"({
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 16,
    "num_attention_heads": 32, "num_key_value_heads": 8, "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-05, "rope_theta": 500000.0, "vocab_size": 128256,
    "tie_word_embeddings": true, "torch_dtype": "bfloat16", "hidden_act": "silu",
    "bos_token_id": 128000, "eos_token_id": [128001, 128008, 128009],
    "rope_scaling": { "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                      "original_max_position_embeddings": 8192, "rope_type": "llama3" } })");
  EXPECT_EQ(c.num_kv_heads, 8);
  EXPECT_EQ(c.head_dim, 64);
  EXPECT_FLOAT_EQ(c.rope_theta, 500000.0f);
  EXPECT_EQ(c.rope_scaling.type, "llama3");
  EXPECT_FLOAT_EQ(c.rope_scaling.factor, 32.0f);
  EXPECT_EQ(c.rope_scaling.original_max_position_embeddings, 8192);
  EXPECT_EQ(c.dtype, DType::BF16);
  EXPECT_EQ(c.eos_token_id, 128001);  // first of the list
  EXPECT_FALSE(c.attention_bias);
}

TEST(ModelConfig, Qwen2ImpliesAttentionBias) {
  const auto c = parse(R"({
    "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "hidden_size": 896,
    "intermediate_size": 4864, "num_hidden_layers": 24, "num_attention_heads": 14,
    "num_key_value_heads": 2, "max_position_embeddings": 32768, "vocab_size": 151936,
    "rope_scaling": null })");
  EXPECT_TRUE(c.attention_bias);
  EXPECT_EQ(c.rope_scaling.type, "");
}

TEST(ModelConfig, MissingKeyThrows) {
  EXPECT_THROW(parse(R"({"model_type": "gpt2", "n_embd": 768})"), std::runtime_error);
}

TEST(ModelConfig, MissingFileThrows) {
  EXPECT_THROW(ModelConfig::from_file("/nonexistent/config.json"), std::runtime_error);
}
