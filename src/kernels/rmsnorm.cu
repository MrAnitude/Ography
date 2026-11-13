// RMSNorm and fused residual-add + RMSNorm (Llama family). One block per row, fp32 statistics.

#include "core/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

namespace {

using namespace detail;

constexpr int kThreads = 256;

template <typename RowT>
__device__ void rms_normalize_row(const RowT* __restrict__ row, const __half* __restrict__ weight,
                                  float eps, int hidden, __half* __restrict__ out, float* scratch) {
  float sq = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = to_float(row[i]);
    sq += v * v;
  }
  const float rstd = rsqrtf(block_sum(sq, scratch) / hidden + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    out[i] = __float2half(to_float(row[i]) * rstd * __half2float(weight[i]));
  }
}

template <typename InT>
__global__ void rmsnorm_kernel(const InT* __restrict__ x, const __half* __restrict__ weight,
                               float eps, int hidden, __half* __restrict__ out) {
  __shared__ float scratch[32];
  const size_t off = static_cast<size_t>(blockIdx.x) * hidden;
  rms_normalize_row(x + off, weight, eps, hidden, out + off, scratch);
}

// x and out may alias: each thread reads x[i] before any thread writes out (block_sum syncs).
__global__ void add_rmsnorm_kernel(const __half* x, float* __restrict__ residual,
                                   const __half* __restrict__ weight, float eps, int hidden,
                                   __half* out) {
  __shared__ float scratch[32];
  const size_t off = static_cast<size_t>(blockIdx.x) * hidden;
  float* res = residual + off;
  // Each thread updates exactly the elements it later reads (same stride pattern).
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) res[i] += __half2float(x[off + i]);
  rms_normalize_row(res, weight, eps, hidden, out + off, scratch);
}

void check_args(const Tensor& x, const Tensor& weight, const Tensor& out, const char* fn) {
  check(x.ndim() == 2, fn, "x must be [T, H]");
  check_cuda(weight, DType::F16, fn, "weight");
  check_cuda(out, DType::F16, fn, "out");
  check(weight.numel() == x.dim(1), fn, "weight must be [H]");
  check(out.shape() == x.shape(), fn, "out must match x");
}

}  // namespace

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out, cudaStream_t stream) {
  const char* fn = "rmsnorm";
  check(x.device() == Device::CUDA && (x.dtype() == DType::F32 || x.dtype() == DType::F16), fn,
        "x must be a CUDA f32 or f16 tensor");
  check_args(x, weight, out, fn);
  const int T = static_cast<int>(x.dim(0)), H = static_cast<int>(x.dim(1));
  if (T == 0) return;
  if (x.dtype() == DType::F32) {
    rmsnorm_kernel<<<T, kThreads, 0, stream>>>(x.data_as<const float>(),
                                               weight.data_as<const __half>(), eps, H,
                                               out.data_as<__half>());
  } else {
    rmsnorm_kernel<<<T, kThreads, 0, stream>>>(x.data_as<const __half>(),
                                               weight.data_as<const __half>(), eps, H,
                                               out.data_as<__half>());
  }
  CUDA_CHECK_LAUNCH();
}

void add_rmsnorm(const Tensor& x, Tensor& residual, const Tensor& weight, float eps, Tensor& out,
                 cudaStream_t stream) {
  const char* fn = "add_rmsnorm";
  check_cuda(x, DType::F16, fn, "x");
  check_cuda(residual, DType::F32, fn, "residual");
  check(residual.shape() == x.shape(), fn, "residual must match x");
  check_args(x, weight, out, fn);
  const int T = static_cast<int>(x.dim(0)), H = static_cast<int>(x.dim(1));
  if (T == 0) return;
  add_rmsnorm_kernel<<<T, kThreads, 0, stream>>>(x.data_as<const __half>(),
                                                 residual.data_as<float>(),
                                                 weight.data_as<const __half>(), eps, H,
                                                 out.data_as<__half>());
  CUDA_CHECK_LAUNCH();
}

}  // namespace ography::kernels
