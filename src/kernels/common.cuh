#pragma once

// Device-side helpers shared by the kernel .cu files. Not part of the public kernels.h API.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <stdexcept>
#include <string>

#include "core/tensor.h"

namespace ography::kernels::detail {

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }

// KV cache element type: fp16, or fp8 E4M3 (no scale; saturates at +-448).
__device__ __forceinline__ float to_float(__nv_fp8_e4m3 v) { return static_cast<float>(v); }
template <>
__device__ __forceinline__ __nv_fp8_e4m3 from_float<__nv_fp8_e4m3>(float v) {
  return __nv_fp8_e4m3(v);  // round to nearest, saturating to the finite range
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffff, v, offset);
  return v;
}

// Sum across the whole block; every thread gets the result. `scratch` needs 32 floats.
// Must be called by all threads of the block.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int n_warps = (blockDim.x + 31) / 32;
  v = warp_sum(v);
  __syncthreads();  // scratch may still be in use by a previous call
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  v = lane < n_warps ? scratch[lane] : 0.0f;
  return warp_sum(v);
}

// Host-side argument checks. Throwing (not aborting) keeps misuse testable.
inline void check(bool ok, const char* fn, const std::string& what) {
  if (!ok) throw std::invalid_argument(std::string(fn) + ": " + what);
}

inline void check_cuda(const Tensor& t, DType dtype, const char* fn, const char* name) {
  check(t.device() == Device::CUDA, fn, std::string(name) + " must be a CUDA tensor");
  check(t.dtype() == dtype, fn,
        std::string(name) + " must be " + std::string(dtype_name(dtype)) + ", got " +
            std::string(dtype_name(t.dtype())));
}

}  // namespace ography::kernels::detail
