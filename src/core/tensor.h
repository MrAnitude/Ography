#pragma once

#include <cstdint>
#include <vector>

#include "core/dtype.h"

namespace ography {

enum class Device : uint8_t { CPU, CUDA };

// A non-owning, contiguous view over memory owned by someone else
// (a DeviceArena, an mmapped safetensors file, a pinned host buffer, ...).
// Tensors are cheap to copy; ownership never lives here.
class Tensor {
 public:
  Tensor() = default;
  Tensor(void* data, DType dtype, std::vector<int64_t> shape, Device device);

  void* data() const { return data_; }
  template <typename T>
  T* data_as() const { return static_cast<T*>(data_); }

  DType dtype() const { return dtype_; }
  Device device() const { return device_; }
  const std::vector<int64_t>& shape() const { return shape_; }
  int ndim() const { return static_cast<int>(shape_.size()); }
  int64_t dim(int i) const;  // supports negative indexing
  int64_t numel() const;
  size_t nbytes() const;
  bool empty() const { return data_ == nullptr; }

  // View of rows [start, start + count) along dim 0 (used for KV cache slices).
  Tensor slice_rows(int64_t start, int64_t count) const;

  // TODO: reshape/view, strides (only if a kernel genuinely needs non-contiguous input),
  //       copy_to(Device) helpers for tests.

 private:
  void* data_ = nullptr;
  DType dtype_ = DType::F32;
  std::vector<int64_t> shape_;  // TODO: small fixed-size array to avoid heap use on the hot path
  Device device_ = Device::CPU;
};

}  // namespace ography
