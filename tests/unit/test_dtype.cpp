#include <gtest/gtest.h>

#include "core/dtype.h"

using namespace ography;

TEST(DType, Bits) {
  EXPECT_EQ(dtype_bits(DType::F32), 32u);
  EXPECT_EQ(dtype_bits(DType::BF16), 16u);
  EXPECT_EQ(dtype_bits(DType::FP8_E4M3), 8u);
  EXPECT_EQ(dtype_bits(DType::I4), 4u);
}

TEST(DType, FromSafetensors) {
  EXPECT_EQ(dtype_from_safetensors("BF16"), DType::BF16);
  EXPECT_EQ(dtype_from_safetensors("F8_E4M3"), DType::FP8_E4M3);
  EXPECT_THROW(dtype_from_safetensors("F64"), std::invalid_argument);
}
