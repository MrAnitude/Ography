// Bias + GELU (tanh approximation, HF "gelu_new"), in place over the MLP up-projection.

#include <algorithm>

#include "core/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

namespace {

using namespace detail;

__device__ __forceinline__ float gelu_tanh(float x) {
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  return 0.5f * x * (1.0f + tanhf(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

__global__ void bias_gelu_kernel(__half* __restrict__ x, const __half* __restrict__ bias,
                                 int64_t n, int cols) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float v = __half2float(x[i]) + __half2float(bias[i % cols]);
    x[i] = __float2half(gelu_tanh(v));
  }
}

}  // namespace

void bias_gelu(Tensor& x, const Tensor& bias, cudaStream_t stream) {
  const char* fn = "bias_gelu";
  check_cuda(x, DType::F16, fn, "x");
  check_cuda(bias, DType::F16, fn, "bias");
  check(x.ndim() == 2 && bias.numel() == x.dim(1), fn, "x must be [T, I], bias [I]");
  const int64_t n = x.numel();
  if (n == 0) return;
  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((n + threads - 1) / threads, 4096));
  bias_gelu_kernel<<<blocks, threads, 0, stream>>>(x.data_as<__half>(), bias.data_as<const __half>(),
                                                   n, static_cast<int>(x.dim(1)));
  CUDA_CHECK_LAUNCH();
}

}  // namespace ography::kernels
