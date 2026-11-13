#include <gtest/gtest.h>

#include <stdexcept>

#include "core/tensor.h"

using namespace ography;

TEST(Tensor, ShapeQueries) {
  Tensor t(nullptr, DType::F16, {4, 3, 2}, Device::CPU);
  EXPECT_EQ(t.ndim(), 3);
  EXPECT_EQ(t.numel(), 24);
  EXPECT_EQ(t.nbytes(), 48u);
  EXPECT_EQ(t.dim(0), 4);
  EXPECT_EQ(t.dim(-1), 2);
  EXPECT_EQ(t.dim(-3), 4);
  EXPECT_THROW(t.dim(3), std::out_of_range);
  EXPECT_THROW(t.dim(-4), std::out_of_range);
}

TEST(Tensor, PackedInt4Bytes) {
  EXPECT_EQ(Tensor(nullptr, DType::I4, {8, 16}, Device::CPU).nbytes(), 64u);
  EXPECT_EQ(Tensor(nullptr, DType::I4, {3}, Device::CPU).nbytes(), 2u);  // rounds up
}

TEST(Tensor, ScalarHasOneElement) {
  Tensor t(nullptr, DType::F32, {}, Device::CPU);
  EXPECT_EQ(t.numel(), 1);
  EXPECT_EQ(t.nbytes(), 4u);
}

TEST(Tensor, SliceRows) {
  float data[6 * 4] = {};
  Tensor t(data, DType::F32, {6, 4}, Device::CPU);
  Tensor s = t.slice_rows(2, 3);
  EXPECT_EQ(s.data_as<float>(), data + 8);
  EXPECT_EQ(s.shape(), (std::vector<int64_t>{3, 4}));
  EXPECT_EQ(s.dtype(), DType::F32);
  EXPECT_EQ(t.slice_rows(6, 0).numel(), 0);
  EXPECT_THROW(t.slice_rows(5, 2), std::out_of_range);
  EXPECT_THROW(t.slice_rows(-1, 1), std::out_of_range);
}
