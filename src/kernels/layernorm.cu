// LayerNorm and fused residual-add + LayerNorm (GPT-2). One block per row, fp32 statistics.
// Mean and variance are computed in two passes ((x - mean)^2, not E[x^2] - mean^2) for accuracy;
// the row is small enough to stay in L1/L2 between passes.

#include "core/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

namespace {

using namespace detail;

constexpr int kThreads = 256;

// Normalizes `row` (already final values) into out. All threads of the block participate.
template <typename RowT>
__device__ void normalize_row(const RowT* __restrict__ row, const __half* __restrict__ weight,
                              const __half* __restrict__ bias, float eps, int hidden,
                              __half* __restrict__ out, float* scratch) {
  float sum = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) sum += to_float(row[i]);
  const float mean = block_sum(sum, scratch) / hidden;

  float sq = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float d = to_float(row[i]) - mean;
    sq += d * d;
  }
  const float rstd = rsqrtf(block_sum(sq, scratch) / hidden + eps);

  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = (to_float(row[i]) - mean) * rstd;
    out[i] = __float2half(v * __half2float(weight[i]) + __half2float(bias[i]));
  }
}

template <typename InT>
__global__ void layernorm_kernel(const InT* __restrict__ x, const __half* __restrict__ weight,
                                 const __half* __restrict__ bias, float eps, int hidden,
                                 __half* __restrict__ out) {
  __shared__ float scratch[32];
  const size_t off = static_cast<size_t>(blockIdx.x) * hidden;
  normalize_row(x + off, weight, bias, eps, hidden, out + off, scratch);
}

__global__ void add_layernorm_kernel(const __half* x, const __half* __restrict__ x_bias,
                                     float* __restrict__ residual,
                                     const __half* __restrict__ weight,
                                     const __half* __restrict__ bias, float eps, int hidden,
                                     __half* out) {
  __shared__ float scratch[32];
  const size_t off = static_cast<size_t>(blockIdx.x) * hidden;
  float* res = residual + off;
  // Each thread updates exactly the elements it later reads in normalize_row (same stride
  // pattern), so no barrier is needed between the update and the statistics pass.
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = res[i] + __half2float(x[off + i]);
    if (x_bias != nullptr) v += __half2float(x_bias[i]);
    res[i] = v;
  }
  // out may alias x: every read of x above happens before this thread's writes below, and
  // other threads only touch their own indices.
  normalize_row(static_cast<const float*>(res), weight, bias, eps, hidden, out + off, scratch);
}

void check_norm_params(const Tensor& weight, const Tensor& bias, int64_t hidden, const char* fn) {
  check_cuda(weight, DType::F16, fn, "weight");
  check_cuda(bias, DType::F16, fn, "bias");
  check(weight.numel() == hidden && bias.numel() == hidden, fn, "weight/bias must be [hidden]");
}

}  // namespace

void layernorm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& out,
               cudaStream_t stream) {
  const char* fn = "layernorm";
  check(x.ndim() == 2, fn, "x must be [T, hidden]");
  check(x.device() == Device::CUDA, fn, "x must be a CUDA tensor");
  check_cuda(out, DType::F16, fn, "out");
  check(out.shape() == x.shape(), fn, "out must match x");
  const int T = static_cast<int>(x.dim(0));
  const int H = static_cast<int>(x.dim(1));
  check_norm_params(weight, bias, H, fn);
  if (T == 0) return;

  if (x.dtype() == DType::F32) {
    layernorm_kernel<float><<<T, kThreads, 0, stream>>>(
        x.data_as<const float>(), weight.data_as<const __half>(), bias.data_as<const __half>(),
        eps, H, out.data_as<__half>());
  } else {
    check(x.dtype() == DType::F16, fn, "x must be f32 or f16");
    layernorm_kernel<__half><<<T, kThreads, 0, stream>>>(
        x.data_as<const __half>(), weight.data_as<const __half>(), bias.data_as<const __half>(),
        eps, H, out.data_as<__half>());
  }
  CUDA_CHECK_LAUNCH();
}

void add_layernorm(const Tensor& x, const Tensor& x_bias, Tensor& residual, const Tensor& weight,
                   const Tensor& bias, float eps, Tensor& out, cudaStream_t stream) {
  const char* fn = "add_layernorm";
  check_cuda(x, DType::F16, fn, "x");
  check_cuda(residual, DType::F32, fn, "residual");
  check_cuda(out, DType::F16, fn, "out");
  check(x.ndim() == 2 && residual.shape() == x.shape() && out.shape() == x.shape(), fn,
        "x, residual and out must all be [T, hidden]");
  const int T = static_cast<int>(x.dim(0));
  const int H = static_cast<int>(x.dim(1));
  check_norm_params(weight, bias, H, fn);
  if (!x_bias.empty()) {
    check_cuda(x_bias, DType::F16, fn, "x_bias");
    check(x_bias.numel() == H, fn, "x_bias must be [hidden]");
  }
  if (T == 0) return;

  add_layernorm_kernel<<<T, kThreads, 0, stream>>>(
      x.data_as<const __half>(), x_bias.empty() ? nullptr : x_bias.data_as<const __half>(),
      residual.data_as<float>(), weight.data_as<const __half>(), bias.data_as<const __half>(),
      eps, H, out.data_as<__half>());
  CUDA_CHECK_LAUNCH();
}

}  // namespace ography::kernels
