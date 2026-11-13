#pragma once

// Helpers for kernel tests: random fp16-representable data, host<->device copies, comparison.

#include <gtest/gtest.h>

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "core/cuda_check.h"
#include "core/memory.h"

namespace ography::test {

// Values rounded through fp16, so the CPU reference sees exactly what the GPU sees.
inline std::vector<float> random_f16_values(size_t n, uint32_t seed, float lo = -1.0f,
                                            float hi = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = __half2float(__float2half(dist(rng)));
  return v;
}

inline Tensor to_device(DeviceArena& arena, const std::vector<float>& v, DType dtype,
                        std::vector<int64_t> shape) {
  Tensor t = arena.allocate(dtype, std::move(shape));
  EXPECT_EQ(static_cast<size_t>(t.numel()), v.size());
  if (dtype == DType::F32) {
    CUDA_CHECK(cudaMemcpy(t.data(), v.data(), t.nbytes(), cudaMemcpyHostToDevice));
  } else {
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
    CUDA_CHECK(cudaMemcpy(t.data(), h.data(), t.nbytes(), cudaMemcpyHostToDevice));
  }
  return t;
}

inline Tensor to_device_i32(DeviceArena& arena, const std::vector<int32_t>& v) {
  Tensor t = arena.allocate(DType::I32, {static_cast<int64_t>(v.size())});
  CUDA_CHECK(cudaMemcpy(t.data(), v.data(), t.nbytes(), cudaMemcpyHostToDevice));
  return t;
}

inline std::vector<float> to_host(const Tensor& t) {
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> out(t.numel());
  if (t.dtype() == DType::F32) {
    CUDA_CHECK(cudaMemcpy(out.data(), t.data(), t.nbytes(), cudaMemcpyDeviceToHost));
  } else {
    std::vector<__half> h(t.numel());
    CUDA_CHECK(cudaMemcpy(h.data(), t.data(), t.nbytes(), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h.size(); ++i) out[i] = __half2float(h[i]);
  }
  return out;
}

inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  EXPECT_EQ(a.size(), b.size());
  float m = 0.0f;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

}  // namespace ography::test
