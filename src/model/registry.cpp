#include "model/registry.h"

#include "core/logging.h"
#include "model/arch/llama.h"

namespace ography {

std::unique_ptr<Model> load_model(const std::string& model_dir, DeviceArena& arena,
                                  cublasHandle_t cublas) {
  // TODO: parse config, open checkpoint, map_weights, then dispatch on architecture:
  //   LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM -> LlamaModel
  OGRAPHY_TODO("load_model");
}

}  // namespace ography
