#include "core/memory.h"

#include "core/cuda_check.h"
#include "core/logging.h"

namespace ography {

DeviceArena::DeviceArena(size_t capacity_bytes) : capacity_(capacity_bytes) {
  OGRAPHY_TODO("DeviceArena: cudaMalloc capacity_bytes");
}

DeviceArena::~DeviceArena() {
  // TODO: cudaFree(base_) once the constructor allocates.
}

Tensor DeviceArena::allocate(DType dtype, std::vector<int64_t> shape) {
  OGRAPHY_TODO("DeviceArena::allocate");
}

void DeviceArena::rewind(size_t mark) {
  OGRAPHY_TODO("DeviceArena::rewind");
}

size_t query_free_device_memory() {
  OGRAPHY_TODO("query_free_device_memory via cudaMemGetInfo");
}

}  // namespace ography
