#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace ography {

// Element types the engine stores. I4 is packed two-per-byte (weight-only quantization).
enum class DType : uint8_t {
  F32,
  F16,
  BF16,
  FP8_E4M3,  // Ada (sm_89) native FP8: weights and/or KV cache
  I8,
  I4,
  I32,
  I64,
};

size_t dtype_bits(DType dtype);
std::string_view dtype_name(DType dtype);

// Parses the dtype strings used in safetensors headers ("F16", "BF16", "F8_E4M3", ...).
// Throws std::invalid_argument for unsupported types.
DType dtype_from_safetensors(std::string_view name);

}  // namespace ography
