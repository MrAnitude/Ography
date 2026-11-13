// Linear layers via cuBLAS. Row-major y = x @ W^T maps to column-major C^T = W x^T, so no
// transposes are materialised:
//   x  row-major [T, in]   == column-major [in, T]   (ld = in)  -> B, op N
//   W  row-major [out, in] == column-major [in, out] (ld = in)  -> A, op T  => [out, in]
//   y  row-major [T, out]  == column-major [out, T]  (ld = out) -> C

#include "core/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

using namespace detail;

void gemm(cublasHandle_t handle, const Tensor& x, const Tensor& weight, Tensor& out,
          cudaStream_t stream) {
  const char* fn = "gemm";
  check_cuda(x, DType::F16, fn, "x");
  check_cuda(weight, DType::F16, fn, "weight");
  check(out.device() == Device::CUDA, fn, "out must be a CUDA tensor");
  check(out.dtype() == DType::F16 || out.dtype() == DType::F32, fn, "out must be f16 or f32");
  check(x.ndim() == 2 && weight.ndim() == 2 && out.ndim() == 2, fn, "all operands must be 2-d");
  const int T = static_cast<int>(x.dim(0));
  const int in = static_cast<int>(x.dim(1));
  const int n_out = static_cast<int>(weight.dim(0));
  check(weight.dim(1) == in, fn, "weight must be [out, in] matching x [T, in]");
  check(out.dim(0) == T && out.dim(1) == n_out, fn, "out must be [T, out]");
  if (T == 0) return;

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cudaDataType_t c_type = out.dtype() == DType::F32 ? CUDA_R_32F : CUDA_R_16F;
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            /*m=*/n_out, /*n=*/T, /*k=*/in, &alpha,
                            weight.data(), CUDA_R_16F, /*lda=*/in,
                            x.data(), CUDA_R_16F, /*ldb=*/in, &beta,
                            out.data(), c_type, /*ldc=*/n_out,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

}  // namespace ography::kernels
