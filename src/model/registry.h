#pragma once

#include <memory>
#include <string>

#include <cublas_v2.h>

#include "core/memory.h"
#include "model/model.h"

namespace ography {

// Loads config.json + safetensors from a HF model directory and builds the right
// Model implementation for config.architecture. max_tokens bounds the tokens per forward
// pass (the prefill chunk), which sizes the model's scratch activations.
std::unique_ptr<Model> load_model(const std::string& model_dir, DeviceArena& arena,
                                  cublasHandle_t cublas, int max_tokens = 512);

}  // namespace ography
