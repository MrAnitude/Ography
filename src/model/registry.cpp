#include "model/registry.h"

#include <filesystem>
#include <stdexcept>

#include "core/logging.h"
#include "model/arch/gpt2.h"
#include "model/arch/llama.h"
#include "model/safetensors.h"
#include "model/weight_map.h"

namespace ography {

std::unique_ptr<Model> load_model(const std::string& model_dir, DeviceArena& arena,
                                  cublasHandle_t cublas, int max_tokens) {
  ModelConfig config =
      ModelConfig::from_file((std::filesystem::path(model_dir) / "config.json").string());
  SafetensorsModel checkpoint(model_dir);

  if (config.model_type == "gpt2") {
    Gpt2Weights weights = map_gpt2_weights(checkpoint, config, arena);
    return std::make_unique<Gpt2Model>(std::move(config), std::move(weights), arena, cublas,
                                       max_tokens);
  }
  if (config.architecture == "LlamaForCausalLM" || config.architecture == "MistralForCausalLM" ||
      config.architecture == "Qwen2ForCausalLM") {
    LlamaWeights weights = map_llama_weights(checkpoint, config, arena);
    return std::make_unique<LlamaModel>(std::move(config), std::move(weights), arena, cublas,
                                        max_tokens);
  }
  throw std::runtime_error(model_dir + ": unsupported architecture \"" + config.architecture +
                           "\" (model_type \"" + config.model_type + "\")");
}

}  // namespace ography
