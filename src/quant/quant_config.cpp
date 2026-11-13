#include "quant/quant_config.h"

#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace ography {

QuantConfig QuantConfig::from_hf_config(const nlohmann::json& config) {
  QuantConfig q;
  const auto it = config.find("quantization_config");
  if (it == config.end() || it->is_null()) return q;
  const nlohmann::json& qc = *it;
  const std::string method = qc.value("quant_method", "");

  if (method == "awq" || method == "gptq") {
    q.method = method == "awq" ? QuantMethod::AWQ : QuantMethod::GPTQ;
    q.bits = qc.value("bits", 4);
    q.group_size = qc.value("group_size", 128);
    if (q.bits != 4) throw std::runtime_error(method + ": only 4-bit weights are supported");
    if (q.group_size <= 0 || q.group_size % 32 != 0) {
      throw std::runtime_error(method + ": group_size must be a positive multiple of 32 "
                               "(per-channel quantization, group_size -1, is not supported)");
    }
    if (q.method == QuantMethod::AWQ) {
      const std::string version = qc.value("version", "gemm");
      if (version != "gemm") throw std::runtime_error("awq: only the \"gemm\" packing is supported");
      q.zero_point = qc.value("zero_point", true);
      if (!q.zero_point) throw std::runtime_error("awq: zero_point false is not supported");
    } else {
      if (qc.value("desc_act", false)) {
        throw std::runtime_error("gptq: act-order (desc_act) checkpoints are not supported");
      }
      q.zero_point = !qc.value("sym", true);
    }
    return q;
  }
  if (method == "fp8" || method == "compressed-tensors") {
    throw std::runtime_error("quantization \"" + method + "\" is not supported yet (use AWQ or GPTQ)");
  }
  throw std::runtime_error("unknown quantization method \"" + method + "\"");
}

}  // namespace ography
