// Llama-family kernels vs CPU references: RMSNorm, fused add + RMSNorm, SwiGLU, and rotary
// embeddings applied by split_qkv.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "kernels/kernels.h"
#include "test_util.h"

using namespace ography;
using namespace ography::test;

namespace {

std::vector<float> cpu_rmsnorm(const std::vector<float>& x, const std::vector<float>& w, int T,
                               int H, float eps) {
  std::vector<float> out(x.size());
  for (int t = 0; t < T; ++t) {
    double sq = 0;
    for (int i = 0; i < H; ++i) sq += double(x[t * H + i]) * x[t * H + i];
    const double rstd = 1.0 / std::sqrt(sq / H + eps);
    for (int i = 0; i < H; ++i) out[t * H + i] = float(x[t * H + i] * rstd) * w[i];
  }
  return out;
}

}  // namespace

TEST(Kernels, RmsNorm) {
  DeviceArena arena(size_t{64} << 20);
  const float eps = 1e-6f;
  for (int H : {896, 1536, 2048, 4096, 8192}) {
    SCOPED_TRACE(H);
    const int T = 5;
    const auto x = random_f16_values(size_t(T) * H, H, -3.f, 3.f);
    const auto w = random_f16_values(H, H + 1, 0.5f, 1.5f);
    const auto want = cpu_rmsnorm(x, w, T, H, eps);
    Tensor d_w = to_device(arena, w, DType::F16, {H});
    for (DType in : {DType::F32, DType::F16}) {
      Tensor d_x = to_device(arena, x, in, {T, H});
      Tensor out = arena.allocate(DType::F16, {T, H});
      kernels::rmsnorm(d_x, d_w, eps, out, nullptr);
      EXPECT_LT(max_abs_diff(to_host(out), want), 1e-2f);
    }
  }
}

TEST(Kernels, AddRmsNorm) {
  DeviceArena arena(size_t{64} << 20);
  const int T = 7, H = 2048;
  const float eps = 1e-5f;
  const auto x = random_f16_values(size_t(T) * H, 1, -1.f, 1.f);
  const auto res = random_f16_values(size_t(T) * H, 2, -4.f, 4.f);
  const auto w = random_f16_values(H, 3, 0.5f, 1.5f);
  std::vector<float> res_want(res.size());
  for (size_t i = 0; i < res.size(); ++i) res_want[i] = res[i] + x[i];
  const auto want = cpu_rmsnorm(res_want, w, T, H, eps);

  Tensor d_x = to_device(arena, x, DType::F16, {T, H});
  Tensor d_res = to_device(arena, res, DType::F32, {T, H});
  Tensor d_w = to_device(arena, w, DType::F16, {H});
  kernels::add_rmsnorm(d_x, d_res, d_w, eps, d_x, nullptr);  // out aliases x
  EXPECT_LT(max_abs_diff(to_host(d_res), res_want), 1e-5f);
  EXPECT_LT(max_abs_diff(to_host(d_x), want), 1e-2f);
}

TEST(Kernels, SwiGlu) {
  DeviceArena arena(size_t{64} << 20);
  const int T = 3, I = 8960;
  const auto gu = random_f16_values(size_t(T) * 2 * I, 4, -6.f, 6.f);
  std::vector<float> want(size_t(T) * I);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < I; ++i) {
      const float g = gu[size_t(t) * 2 * I + i], u = gu[size_t(t) * 2 * I + I + i];
      want[size_t(t) * I + i] = g / (1.f + std::exp(-g)) * u;
    }
  Tensor d_gu = to_device(arena, gu, DType::F16, {T, 2 * I});
  Tensor out = arena.allocate(DType::F16, {T, I});
  kernels::swiglu(d_gu, out, nullptr);
  EXPECT_LT(max_abs_diff(to_host(out), want), 3e-2f);  // |out| up to ~36: fp16 rounding
}

// split_qkv with rope: HF rotate_half on q and k, written to q and the cache at each
// position; v untouched. Positions up to 120K check the angle reduction stays accurate.
TEST(Kernels, SplitQkvAppliesRope) {
  DeviceArena arena(size_t{128} << 20);  // two 131K-position caches are 64 MiB
  const int nh = 4, nkv = 2, hd = 64, half = hd / 2, max_seq = 131072;
  const std::vector<int32_t> pos = {0, 1, 17, 4095, 120000};
  const int T = static_cast<int>(pos.size());
  const int width = (nh + 2 * nkv) * hd;
  const auto qkv = random_f16_values(size_t(T) * width, 5);
  const auto bias = random_f16_values(width, 6, -0.5f, 0.5f);
  std::vector<float> inv(half);
  for (int i = 0; i < half; ++i) inv[i] = 1.0f / std::pow(500000.0f, float(2 * i) / hd);

  // Expected q and k rows (bias added in fp16, then rotated in double precision).
  auto rotated = [&](int t, int head) {
    std::vector<float> out(hd);
    for (int i = 0; i < half; ++i) {
      const int j = head * hd + i;
      const float x1 = __half2float(__float2half(qkv[t * width + j] + bias[j]));
      const float x2 = __half2float(__float2half(qkv[t * width + j + half] + bias[j + half]));
      const double a = double(float(pos[t]) * inv[i]);
      out[i] = float(x1 * std::cos(a) - x2 * std::sin(a));
      out[i + half] = float(x2 * std::cos(a) + x1 * std::sin(a));
    }
    return out;
  };

  Tensor d_qkv = to_device(arena, qkv, DType::F16, {T, width});
  Tensor d_bias = to_device(arena, bias, DType::F16, {width});
  Tensor d_pos = to_device_i32(arena, pos);
  Tensor d_inv = to_device(arena, inv, DType::F32, {half});
  Tensor q = arena.allocate(DType::F16, {T, nh, hd});
  Tensor k = arena.allocate(DType::F16, {max_seq, nkv, hd});
  Tensor v = arena.allocate(DType::F16, {max_seq, nkv, hd});
  kernels::split_qkv(d_qkv, d_bias, nh, nkv, q, k, v, d_pos, d_inv, nullptr);

  const auto hq = to_host(q);
  float max_err = 0;
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < nh; ++h) {
      const auto want = rotated(t, h);
      for (int d = 0; d < hd; ++d) max_err = std::max(max_err, std::fabs(hq[(t * nh + h) * hd + d] - want[d]));
    }
    const auto hk = to_host(k.slice_rows(pos[t], 1));
    const auto hv = to_host(v.slice_rows(pos[t], 1));
    for (int h = 0; h < nkv; ++h) {
      const auto want = rotated(t, nh + h);
      for (int d = 0; d < hd; ++d) {
        max_err = std::max(max_err, std::fabs(hk[h * hd + d] - want[d]));
        const int j = (nh + nkv + h) * hd + d;
        EXPECT_EQ(hv[h * hd + d], __half2float(__float2half(qkv[t * width + j] + bias[j])));
      }
    }
  }
  EXPECT_LT(max_err, 4e-3f);  // fp16 output of values up to ~1.5
}
