// Embedding, LayerNorm, add+LayerNorm, bias+GELU and GEMM against CPU fp32 references.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "kernels/kernels.h"
#include "test_util.h"

using namespace ography;
using namespace ography::test;

namespace {

std::vector<float> cpu_layernorm(const std::vector<float>& x, const std::vector<float>& w,
                                 const std::vector<float>& b, int T, int H, float eps) {
  std::vector<float> out(x.size());
  for (int t = 0; t < T; ++t) {
    const float* r = &x[static_cast<size_t>(t) * H];
    double mean = 0, var = 0;
    for (int i = 0; i < H; ++i) mean += r[i];
    mean /= H;
    for (int i = 0; i < H; ++i) var += (r[i] - mean) * (r[i] - mean);
    var /= H;
    const double rstd = 1.0 / std::sqrt(var + eps);
    for (int i = 0; i < H; ++i) out[static_cast<size_t>(t) * H + i] = static_cast<float>((r[i] - mean) * rstd * w[i] + b[i]);
  }
  return out;
}

float cpu_gelu(float x) {
  return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

class KernelTest : public ::testing::Test {
 protected:
  DeviceArena arena_{size_t{256} << 20};
  cudaStream_t stream_ = nullptr;  // default stream
};

}  // namespace

TEST_F(KernelTest, EmbeddingWithPositions) {
  const int V = 50, P = 16, H = 96, T = 5;
  auto wte_h = random_f16_values(V * H, 1);
  auto wpe_h = random_f16_values(P * H, 2);
  std::vector<int32_t> ids = {0, 7, 49, 7, 3};
  std::vector<int32_t> pos = {4, 5, 6, 7, 15};
  Tensor wte = to_device(arena_, wte_h, DType::F16, {V, H});
  Tensor wpe = to_device(arena_, wpe_h, DType::F16, {P, H});
  Tensor d_ids = to_device_i32(arena_, ids);
  Tensor d_pos = to_device_i32(arena_, pos);
  Tensor out = arena_.allocate(DType::F32, {T, H});

  kernels::embedding_with_positions(wte, wpe, d_ids, d_pos, out, stream_);
  auto got = to_host(out);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < H; ++i)
      ASSERT_FLOAT_EQ(got[t * H + i], wte_h[ids[t] * H + i] + wpe_h[pos[t] * H + i]) << t << "," << i;

  // Plain embedding into fp16.
  Tensor out16 = arena_.allocate(DType::F16, {T, H});
  kernels::embedding(wte, d_ids, out16, stream_);
  auto got16 = to_host(out16);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < H; ++i) ASSERT_EQ(got16[t * H + i], wte_h[ids[t] * H + i]);
}

TEST_F(KernelTest, LayerNormFromF32AndF16) {
  const float eps = 1e-5f;
  for (int H : {64, 768, 1000, 4096}) {
    const int T = 7;
    auto x_h = random_f16_values(T * H, 10 + H, -3.0f, 5.0f);  // non-zero mean on purpose
    auto w_h = random_f16_values(H, 11, 0.5f, 1.5f);
    auto b_h = random_f16_values(H, 12, -0.2f, 0.2f);
    auto expect = cpu_layernorm(x_h, w_h, b_h, T, H, eps);
    Tensor w = to_device(arena_, w_h, DType::F16, {H});
    Tensor b = to_device(arena_, b_h, DType::F16, {H});
    Tensor out = arena_.allocate(DType::F16, {T, H});
    for (DType in_dtype : {DType::F32, DType::F16}) {
      Tensor x = to_device(arena_, x_h, in_dtype, {T, H});
      kernels::layernorm(x, w, b, eps, out, stream_);
      EXPECT_LT(max_abs_diff(to_host(out), expect), 1e-2f) << "H=" << H;
    }
  }
}

TEST_F(KernelTest, AddLayerNormUpdatesResidual) {
  const int T = 4, H = 768;
  const float eps = 1e-5f;
  auto x_h = random_f16_values(T * H, 20);
  auto xb_h = random_f16_values(H, 21, -0.5f, 0.5f);
  auto res_h = random_f16_values(T * H, 22, -4.0f, 4.0f);
  auto w_h = random_f16_values(H, 23, 0.5f, 1.5f);
  auto b_h = random_f16_values(H, 24, -0.2f, 0.2f);

  std::vector<float> new_res(T * H);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < H; ++i) new_res[t * H + i] = res_h[t * H + i] + x_h[t * H + i] + xb_h[i];
  auto expect = cpu_layernorm(new_res, w_h, b_h, T, H, eps);

  Tensor x = to_device(arena_, x_h, DType::F16, {T, H});
  Tensor xb = to_device(arena_, xb_h, DType::F16, {H});
  Tensor res = to_device(arena_, res_h, DType::F32, {T, H});
  Tensor w = to_device(arena_, w_h, DType::F16, {H});
  Tensor b = to_device(arena_, b_h, DType::F16, {H});

  // out aliases x: the documented in-place use.
  kernels::add_layernorm(x, xb, res, w, b, eps, x, stream_);
  EXPECT_LT(max_abs_diff(to_host(res), new_res), 1e-5f);
  EXPECT_LT(max_abs_diff(to_host(x), expect), 1e-2f);

  // Without bias.
  Tensor res2 = to_device(arena_, res_h, DType::F32, {T, H});
  Tensor x2 = to_device(arena_, x_h, DType::F16, {T, H});
  Tensor out2 = arena_.allocate(DType::F16, {T, H});
  kernels::add_layernorm(x2, Tensor(), res2, w, b, eps, out2, stream_);
  auto r2 = to_host(res2);
  EXPECT_FLOAT_EQ(r2[5], res_h[5] + x_h[5]);
}

TEST_F(KernelTest, BiasGelu) {
  const int T = 3, I = 3072;
  auto x_h = random_f16_values(T * I, 30, -6.0f, 6.0f);
  auto b_h = random_f16_values(I, 31, -0.5f, 0.5f);
  std::vector<float> expect(T * I);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < I; ++i) expect[t * I + i] = cpu_gelu(x_h[t * I + i] + b_h[i]);
  Tensor x = to_device(arena_, x_h, DType::F16, {T, I});
  Tensor b = to_device(arena_, b_h, DType::F16, {I});
  kernels::bias_gelu(x, b, stream_);
  EXPECT_LT(max_abs_diff(to_host(x), expect), 1e-2f);
}

TEST_F(KernelTest, GemmF16AndF32Out) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  for (auto [T, in, out_f] : {std::tuple{1, 768, 2304}, {5, 768, 3072}, {33, 3072, 768}, {1, 64, 50257}}) {
    auto x_h = random_f16_values(T * in, 40 + T);
    auto w_h = random_f16_values(static_cast<size_t>(out_f) * in, 41 + T, -0.1f, 0.1f);
    std::vector<float> expect(static_cast<size_t>(T) * out_f);
    for (int t = 0; t < T; ++t)
      for (int o = 0; o < out_f; ++o) {
        double acc = 0;
        for (int k = 0; k < in; ++k) acc += double(x_h[t * in + k]) * w_h[static_cast<size_t>(o) * in + k];
        expect[static_cast<size_t>(t) * out_f + o] = static_cast<float>(acc);
      }
    Tensor x = to_device(arena_, x_h, DType::F16, {T, in});
    Tensor w = to_device(arena_, w_h, DType::F16, {out_f, in});
    Tensor y32 = arena_.allocate(DType::F32, {T, out_f});
    Tensor y16 = arena_.allocate(DType::F16, {T, out_f});
    kernels::gemm(handle, x, w, y32, stream_);
    kernels::gemm(handle, x, w, y16, stream_);
    EXPECT_LT(max_abs_diff(to_host(y32), expect), 2e-3f) << T << "x" << in << "->" << out_f;
    EXPECT_LT(max_abs_diff(to_host(y16), expect), 1e-2f) << T << "x" << in << "->" << out_f;
  }
  // Shape mismatch is reported, not silently computed.
  Tensor x = arena_.allocate(DType::F16, {2, 10});
  Tensor w = arena_.allocate(DType::F16, {4, 11});
  Tensor y = arena_.allocate(DType::F16, {2, 4});
  EXPECT_THROW(kernels::gemm(handle, x, w, y, stream_), std::invalid_argument);
  CUBLAS_CHECK(cublasDestroy(handle));
}
