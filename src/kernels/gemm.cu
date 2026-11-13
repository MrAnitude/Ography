// Linear layers via cuBLAS (cublasGemmEx / cublasLt). Row-major y = x @ W^T maps to
// column-major C^T = W x^T, so no transposes are materialised.

#include "kernels/kernels.h"
#include "core/logging.h"

namespace ography::kernels {

void gemm(cublasHandle_t handle, const Tensor& x, const Tensor& weight, Tensor& out,
          cudaStream_t stream) {
  OGRAPHY_TODO("kernels::gemm (cuBLAS)");
}

}  // namespace ography::kernels
