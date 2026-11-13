#include "core/tensor.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ography {

Tensor::Tensor(void* data, DType dtype, std::vector<int64_t> shape, Device device)
    : data_(data), dtype_(dtype), shape_(std::move(shape)), device_(device) {}

int64_t Tensor::dim(int i) const {
  const int n = ndim();
  const int idx = i < 0 ? i + n : i;
  if (idx < 0 || idx >= n) {
    throw std::out_of_range("Tensor::dim(" + std::to_string(i) + ") on a " + std::to_string(n) +
                            "-d tensor");
  }
  return shape_[idx];
}

int64_t Tensor::numel() const {
  int64_t n = 1;
  for (int64_t d : shape_) n *= d;
  return n;
}

size_t Tensor::nbytes() const {
  // Round up so packed sub-byte types (I4) with an odd element count still cover the tail.
  return (static_cast<size_t>(numel()) * dtype_bits(dtype_) + 7) / 8;
}

Tensor Tensor::slice_rows(int64_t start, int64_t count) const {
  if (ndim() < 1) throw std::invalid_argument("Tensor::slice_rows on a 0-d tensor");
  if (start < 0 || count < 0 || start + count > shape_[0]) {
    throw std::out_of_range("Tensor::slice_rows [" + std::to_string(start) + ", " +
                            std::to_string(start + count) + ") out of " +
                            std::to_string(shape_[0]) + " rows");
  }
  const size_t row_bits = static_cast<size_t>(numel() / std::max<int64_t>(shape_[0], 1)) *
                          dtype_bits(dtype_);
  if (row_bits % 8 != 0) {
    throw std::invalid_argument("Tensor::slice_rows: rows are not byte-aligned");
  }
  std::vector<int64_t> shape = shape_;
  shape[0] = count;
  auto* base = static_cast<char*>(data_) + static_cast<size_t>(start) * (row_bits / 8);
  return Tensor(base, dtype_, std::move(shape), device_);
}

}  // namespace ography
