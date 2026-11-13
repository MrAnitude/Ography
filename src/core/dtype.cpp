#include "core/dtype.h"

#include <stdexcept>
#include <string>

namespace ography {

size_t dtype_bits(DType dtype) {
  switch (dtype) {
    case DType::F32: return 32;
    case DType::F16: return 16;
    case DType::BF16: return 16;
    case DType::FP8_E4M3: return 8;
    case DType::I8: return 8;
    case DType::I4: return 4;
    case DType::I32: return 32;
    case DType::I64: return 64;
  }
  return 0;
}

std::string_view dtype_name(DType dtype) {
  switch (dtype) {
    case DType::F32: return "f32";
    case DType::F16: return "f16";
    case DType::BF16: return "bf16";
    case DType::FP8_E4M3: return "fp8_e4m3";
    case DType::I8: return "i8";
    case DType::I4: return "i4";
    case DType::I32: return "i32";
    case DType::I64: return "i64";
  }
  return "unknown";
}

DType dtype_from_safetensors(std::string_view name) {
  if (name == "F32") return DType::F32;
  if (name == "F16") return DType::F16;
  if (name == "BF16") return DType::BF16;
  if (name == "F8_E4M3") return DType::FP8_E4M3;
  if (name == "I8") return DType::I8;
  if (name == "I32") return DType::I32;
  if (name == "I64") return DType::I64;
  throw std::invalid_argument("unsupported safetensors dtype: " + std::string(name));
}

}  // namespace ography
