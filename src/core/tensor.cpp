#include "core/tensor.h"

#include "core/logging.h"

namespace ography {

Tensor::Tensor(void* data, DType dtype, std::vector<int64_t> shape, Device device)
    : data_(data), dtype_(dtype), shape_(std::move(shape)), device_(device) {}

int64_t Tensor::dim(int i) const {
  OGRAPHY_TODO("Tensor::dim");
}

int64_t Tensor::numel() const {
  OGRAPHY_TODO("Tensor::numel");
}

size_t Tensor::nbytes() const {
  OGRAPHY_TODO("Tensor::nbytes (remember I4 packs two values per byte)");
}

Tensor Tensor::slice_rows(int64_t start, int64_t count) const {
  OGRAPHY_TODO("Tensor::slice_rows");
}

}  // namespace ography
