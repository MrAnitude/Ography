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
  bool zero_point = true;  // false for symmetric GPTQ (zero fixed at 8); both load the same way

  // Throws for formats the engine can't run (FP8 weights, act-order GPTQ, 8-bit, ...).
  static QuantConfig from_hf_config(const nlohmann::json& config);
};

// A linear layer's weights as the engine stores them on the GPU.
// QuantMethod::None: `weight` is fp16 [out, in].
// AWQ / GPTQ (both converted at load time to one layout, see kernels::repack_int4):
//   `weight` is packed int4 [out, in/8] i32, `scales` / `zeros` are [out, in/group] f16.
struct QuantizedWeight {
  Tensor weight;
  Tensor scales;
  Tensor zeros;
  QuantMethod method = QuantMethod::None;
  int group_size = 0;
};

}  // namespace ography
