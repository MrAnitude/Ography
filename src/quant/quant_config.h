#pragma once

#include <nlohmann/json_fwd.hpp>

#include "core/tensor.h"

namespace ography {

enum class QuantMethod { None, FP8, AWQ, GPTQ };

// Parsed from the "quantization_config" section of a HF config.json.
struct QuantConfig {
  QuantMethod method = QuantMethod::None;
  int bits = 16;
  int group_size = 128;
  bool zero_point = true;

  static QuantConfig from_hf_config(const nlohmann::json& config);
};

// A linear layer's weights in whatever form the checkpoint stores them.
// For QuantMethod::None only `weight` is set.
struct QuantizedWeight {
  Tensor weight;   // fp16/bf16/fp8 weight, or packed int4 qweight
  Tensor scales;
  Tensor zeros;
  QuantMethod method = QuantMethod::None;
  int group_size = 0;
};

}  // namespace ography
