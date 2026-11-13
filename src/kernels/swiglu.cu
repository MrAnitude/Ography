// SwiGLU activation: silu(gate) * up over the fused gate_up projection output.

#include "core/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

namespace {

using namespace detail;

__global__ void swiglu_kernel(const __half* __restrict__ gate_up, __half* __restrict__ out,
                              int inter, size_t total) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const size_t t = idx / inter;
  const int i = static_cast<int>(idx % inter);
  const __half* row = gate_up + t * 2 * inter;
  const float g = __half2float(row[i]);
  const float u = __half2float(row[inter + i]);
  out[idx] = __float2half(g / (1.0f + __expf(-g)) * u);
}

}  // namespace

void swiglu(const Tensor& gate_up, Tensor& out, cudaStream_t stream) {
  const char* fn = "swiglu";
  check_cuda(gate_up, DType::F16, fn, "gate_up");
  check_cuda(out, DType::F16, fn, "out");
  check(gate_up.ndim() == 2 && out.ndim() == 2 && gate_up.dim(0) == out.dim(0) &&
            gate_up.dim(1) == 2 * out.dim(1),
        fn, "gate_up must be [T, 2*I] and out [T, I]");
  const size_t total = static_cast<size_t>(out.numel());
  if (total == 0) return;
  constexpr int kThreads = 256;
  swiglu_kernel<<<static_cast<unsigned>((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      gate_up.data_as<const __half>(), out.data_as<__half>(), static_cast<int>(out.dim(1)), total);
  CUDA_CHECK_LAUNCH();
}

}  // namespace ography::kernels
