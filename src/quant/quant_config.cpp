#include "quant/quant_config.h"

#include <nlohmann/json.hpp>

#include "core/logging.h"

namespace ography {

QuantConfig QuantConfig::from_hf_config(const nlohmann::json& config) {
  if (!config.contains("quantization_config")) return QuantConfig{};
  OGRAPHY_TODO("QuantConfig::from_hf_config (quant_method: fp8 / awq / gptq)");
}

}  // namespace ography
