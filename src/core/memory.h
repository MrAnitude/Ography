#pragma once

#include <cstddef>
#include <vector>

#include "core/tensor.h"

namespace ography {

// One big cudaMalloc at startup, then bump allocation. Nothing is freed individually;
// this keeps allocation off the hot path and makes memory use predictable on an 8 GB card.
//
// Typical layout: [weights | KV cache | activation scratch].
// Scratch is reused every forward pass via mark()/rewind().
class DeviceArena {
 public:
  explicit DeviceArena(size_t capacity_bytes);
  ~DeviceArena();
  DeviceArena(const DeviceArena&) = delete;
  DeviceArena& operator=(const DeviceArena&) = delete;

  // Allocates a tensor (256-byte aligned). Aborts if the arena is exhausted.
  Tensor allocate(DType dtype, std::vector<int64_t> shape);

  size_t mark() const { return offset_; }
  void rewind(size_t mark);

  size_t used() const { return offset_; }
  size_t capacity() const { return capacity_; }

 private:
  void* base_ = nullptr;
  size_t capacity_ = 0;
  size_t offset_ = 0;
};

// Returns free device memory, for sizing the arena / KV cache at startup.
size_t query_free_device_memory();

// TODO: PinnedHostBuffer for fast H2D uploads during weight loading.

}  // namespace ography
