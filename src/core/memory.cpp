#include "core/memory.h"

#include <string>

#include "core/cuda_check.h"
#include "core/logging.h"

namespace ography {

namespace {
constexpr size_t kAlignment = 256;

size_t align_up(size_t n) { return (n + kAlignment - 1) / kAlignment * kAlignment; }
}  // namespace

DeviceArena::DeviceArena(size_t capacity_bytes) : capacity_(capacity_bytes) {
  CUDA_CHECK(cudaMalloc(&base_, capacity_bytes));
}

DeviceArena::~DeviceArena() {
  if (base_ != nullptr) cudaFree(base_);
}

Tensor DeviceArena::allocate(DType dtype, std::vector<int64_t> shape) {
  Tensor t(nullptr, dtype, std::move(shape), Device::CUDA);
  const size_t start = align_up(offset_);
  const size_t bytes = t.nbytes();
  if (start + bytes > capacity_) {
    log(LogLevel::Error, "DeviceArena exhausted: need " + std::to_string(bytes) + " bytes at " +
                             std::to_string(start) + ", capacity " + std::to_string(capacity_));
    std::abort();
  }
  offset_ = start + bytes;
  return Tensor(static_cast<char*>(base_) + start, dtype, t.shape(), Device::CUDA);
}

void DeviceArena::rewind(size_t mark) {
  if (mark > offset_) {
    log(LogLevel::Error, "DeviceArena::rewind to " + std::to_string(mark) +
                             " is past the current offset " + std::to_string(offset_));
    std::abort();
  }
  offset_ = mark;
}

size_t query_free_device_memory() {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  return free_bytes;
}

}  // namespace ography
