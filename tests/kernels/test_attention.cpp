// split_qkv + causal attention over the KV cache, against a CPU softmax(QK^T * scale) V.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "kernels/attention.cuh"
#include "kernels/kernels.h"
#include "test_util.h"

using namespace ography;
using namespace ography::test;

namespace {

struct Shape {
  int n_heads, n_kv_heads, head_dim;
};

// q [T, nh, hd]; caches [max_seq, nkv, hd]; query t sees keys [0, positions[t]].
std::vector<float> cpu_attention(const std::vector<float>& q, const std::vector<float>& k,
                                 const std::vector<float>& v, const std::vector<int32_t>& positions,
                                 Shape s, float scale) {
  const int T = static_cast<int>(positions.size());
  std::vector<float> out(static_cast<size_t>(T) * s.n_heads * s.head_dim);
  const int group = s.n_heads / s.n_kv_heads;
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < s.n_heads; ++h) {
      const int kvh = h / group;
      const int n = positions[t] + 1;
      std::vector<double> sc(n);
      double mx = -std::numeric_limits<double>::infinity();
      for (int j = 0; j < n; ++j) {
        double dot = 0;
        for (int d = 0; d < s.head_dim; ++d)
          dot += double(q[(static_cast<size_t>(t) * s.n_heads + h) * s.head_dim + d]) *
                 k[(static_cast<size_t>(j) * s.n_kv_heads + kvh) * s.head_dim + d];
        sc[j] = dot * scale;
        mx = std::max(mx, sc[j]);
      }
      double sum = 0;
      for (auto& x : sc) sum += (x = std::exp(x - mx));
      for (int d = 0; d < s.head_dim; ++d) {
        double o = 0;
        for (int j = 0; j < n; ++j)
          o += sc[j] * v[(static_cast<size_t>(j) * s.n_kv_heads + kvh) * s.head_dim + d];
        out[(static_cast<size_t>(t) * s.n_heads + h) * s.head_dim + d] = static_cast<float>(o / sum);
      }
    }
  }
  return out;
}

class AttentionTest : public ::testing::TestWithParam<Shape> {
 protected:
  DeviceArena arena_{size_t{256} << 20};
};

}  // namespace

// Full path: fused qkv (+bias) -> split_qkv into cache -> attention. Fills the cache in two
// chunks (a prefix, then a continuation at start_pos > 0) and checks the continuation.
TEST_P(AttentionTest, PrefillChunkAtOffset) {
  const Shape s = GetParam();
  const int max_seq = 64, prefix = 19, T = 13;
  const int qw = s.n_heads * s.head_dim, kvw = s.n_kv_heads * s.head_dim, row = qw + 2 * kvw;
  const float scale = 1.0f / std::sqrt(float(s.head_dim));

  Tensor k_cache = arena_.allocate(DType::F16, {max_seq, s.n_kv_heads, s.head_dim});
  Tensor v_cache = arena_.allocate(DType::F16, {max_seq, s.n_kv_heads, s.head_dim});
  auto bias_h = random_f16_values(row, 99, -0.25f, 0.25f);
  Tensor bias = to_device(arena_, bias_h, DType::F16, {row});

  std::vector<float> k_ref(static_cast<size_t>(max_seq) * kvw), v_ref(k_ref.size());
  std::vector<float> q_last;
  std::vector<int32_t> pos_last;
  for (auto [start, n] : {std::pair{0, prefix}, {prefix, T}}) {
    auto qkv_h = random_f16_values(static_cast<size_t>(n) * row, 7 + start, -2.0f, 2.0f);
    std::vector<int32_t> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = start + i;
    Tensor qkv = to_device(arena_, qkv_h, DType::F16, {n, row});
    Tensor d_pos = to_device_i32(arena_, pos);
    Tensor q = arena_.allocate(DType::F16, {n, s.n_heads, s.head_dim});
    kernels::split_qkv(qkv, bias, s.n_heads, s.n_kv_heads, q, k_cache, v_cache, d_pos, Tensor(),
                       nullptr);

    // Reference split: (value + bias) rounded to fp16, as the kernel does.
    auto r16 = [](float x) { return __half2float(__float2half(x)); };
    q_last.assign(static_cast<size_t>(n) * qw, 0);
    for (int t = 0; t < n; ++t)
      for (int j = 0; j < row; ++j) {
        const float val = r16(qkv_h[static_cast<size_t>(t) * row + j] + bias_h[j]);
        if (j < qw) q_last[static_cast<size_t>(t) * qw + j] = val;
        else if (j < qw + kvw) k_ref[static_cast<size_t>(pos[t]) * kvw + j - qw] = val;
        else v_ref[static_cast<size_t>(pos[t]) * kvw + j - qw - kvw] = val;
      }
    EXPECT_EQ(to_host(q), q_last);

    if (start == prefix) {
      pos_last = pos;
      Tensor out = arena_.allocate(DType::F16, {n, s.n_heads, s.head_dim});
      kernels::attention_prefill(q, k_cache, v_cache, d_pos, scale, out, nullptr);
      auto expect = cpu_attention(q_last, k_ref, v_ref, pos, s, scale);
      EXPECT_LT(max_abs_diff(to_host(out), expect), 5e-3f);
    }
  }
  auto kc = to_host(k_cache);
  kc.resize(static_cast<size_t>(prefix + T) * kvw);
  std::vector<float> kr(k_ref.begin(), k_ref.begin() + (prefix + T) * kvw);
  EXPECT_EQ(kc, kr);
}

// Decode: one query against caches of length 1 to 1000 (one or several splits, and a long
// sequence that stresses the online softmax).
TEST_P(AttentionTest, DecodeVariousLengths) {
  const Shape s = GetParam();
  const int max_seq = 1024;
  const float scale = 1.0f / std::sqrt(float(s.head_dim));
  auto k_h = random_f16_values(static_cast<size_t>(max_seq) * s.n_kv_heads * s.head_dim, 3, -2.0f, 2.0f);
  auto v_h = random_f16_values(k_h.size(), 4);
  Tensor k_cache = to_device(arena_, k_h, DType::F16, {max_seq, s.n_kv_heads, s.head_dim});
  Tensor v_cache = to_device(arena_, v_h, DType::F16, {max_seq, s.n_kv_heads, s.head_dim});

  Tensor scratch = arena_.allocate(
      DType::F32,
      {int64_t(kernels::attention_decode_scratch_floats(s.n_heads, s.head_dim, max_seq))});
  // 1, 17: one split; 129, 1000: several (128-key splits at this capacity).
  for (int seq_len : {1, 17, 129, 1000}) {
    auto q_h = random_f16_values(static_cast<size_t>(s.n_heads) * s.head_dim, 5 + seq_len, -2.0f, 2.0f);
    std::vector<int32_t> pos = {seq_len - 1};
    Tensor q = to_device(arena_, q_h, DType::F16, {1, s.n_heads, s.head_dim});
    Tensor d_pos = to_device_i32(arena_, pos);
    Tensor out = arena_.allocate(DType::F16, {1, s.n_heads, s.head_dim});
    kernels::attention_decode(q, k_cache, v_cache, d_pos, scale, out, scratch, nullptr);
    auto expect = cpu_attention(q_h, k_h, v_h, pos, s, scale);
    EXPECT_LT(max_abs_diff(to_host(out), expect), 5e-3f) << "seq_len=" << seq_len;
  }
}

INSTANTIATE_TEST_SUITE_P(Shapes, AttentionTest,
                         ::testing::Values(Shape{12, 12, 64},   // GPT-2 small
                                           Shape{8, 2, 128},    // GQA 4:1, Llama head_dim
                                           Shape{16, 2, 64},    // GQA 8:1
                                           Shape{4, 4, 96}),    // head_dim not a power of 2
                         [](const auto& info) {
                           const Shape& s = info.param;
                           return "h" + std::to_string(s.n_heads) + "_kv" +
                                  std::to_string(s.n_kv_heads) + "_d" + std::to_string(s.head_dim);
                         });

// The tensor-core flash kernel against the naive kernel (validated against the CPU above), on
// shapes that span several query and key tiles, partial tiles, offsets, scattered positions,
// and both cache types.
TEST(FlashAttention, MatchesNaive) {
  DeviceArena arena(size_t{256} << 20);
  struct Case { int n_heads, n_kv_heads, head_dim, T, offset; bool scattered; };
  const Case cases[] = {
      {28, 4, 128, 200, 0, false},   // Qwen2.5 7B, from the start
      {28, 4, 128, 77, 301, false},  // continuation chunk at an odd offset
      {32, 8, 64, 130, 1000, false}, // Llama 3.2 1B
      {12, 12, 64, 1, 50, false},    // a single row
      {8, 2, 128, 90, 0, true},      // positions out of order / with gaps
  };
  for (DType cache_dtype : {DType::F16, DType::FP8_E4M3}) {
    for (const Case& c : cases) {
      SCOPED_TRACE(std::string(dtype_name(cache_dtype)) + " h" + std::to_string(c.n_heads) +
                   " d" + std::to_string(c.head_dim) + " T" + std::to_string(c.T) + " off" +
                   std::to_string(c.offset));
      const size_t mark = arena.mark();
      const int max_seq = 2048;
      std::vector<int32_t> pos(c.T);
      for (int i = 0; i < c.T; ++i) pos[i] = c.scattered ? (i * 37) % 1500 : c.offset + i;
      // Fill the cache through split_qkv so fp8 rounding happens as in the engine.
      const int width = (c.n_heads + 2 * c.n_kv_heads) * c.head_dim;
      std::vector<int32_t> all_pos(max_seq);
      for (int i = 0; i < max_seq; ++i) all_pos[i] = i;
      Tensor k = arena.allocate(cache_dtype, {max_seq, c.n_kv_heads, c.head_dim});
      Tensor v = arena.allocate(cache_dtype, {max_seq, c.n_kv_heads, c.head_dim});
      Tensor qkv_all = to_device(arena, random_f16_values(size_t(max_seq) * width, 3, -2.f, 2.f),
                                 DType::F16, {max_seq, width});
      Tensor q_all = arena.allocate(DType::F16, {max_seq, c.n_heads, c.head_dim});
      kernels::split_qkv(qkv_all, Tensor(), c.n_heads, c.n_kv_heads, q_all, k, v,
                         to_device_i32(arena, all_pos), Tensor(), nullptr);

      Tensor q = to_device(arena, random_f16_values(size_t(c.T) * c.n_heads * c.head_dim, 4, -2.f, 2.f),
                           DType::F16, {c.T, c.n_heads, c.head_dim});
      Tensor d_pos = to_device_i32(arena, pos);
      Tensor flash = arena.allocate(DType::F16, {c.T, c.n_heads, c.head_dim});
      Tensor naive = arena.allocate(DType::F16, {c.T, c.n_heads, c.head_dim});
      const float scale = 1.0f / std::sqrt(float(c.head_dim));
      ASSERT_TRUE(kernels::detail::attention_flash(q, k, v, d_pos, scale, flash, nullptr));
      kernels::detail::attention_naive(q, k, v, d_pos, scale, naive, nullptr, "test");
      EXPECT_LT(max_abs_diff(to_host(flash), to_host(naive)), 4e-3f);
      arena.rewind(mark);
    }
  }
}

// Split-K decode against the naive kernel: model shapes, a large capacity (so splits are not
// the 128-key minimum), lengths from 1 to the last row, both cache types.
TEST(SplitDecode, MatchesNaive) {
  DeviceArena arena(size_t{512} << 20);
  struct Shape { int n_heads, n_kv_heads, head_dim; };
  const int max_seq = 40000;
  for (DType cache_dtype : {DType::F16, DType::FP8_E4M3}) {
    for (const Shape s : {Shape{28, 4, 128}, Shape{32, 8, 64}, Shape{12, 2, 128}, Shape{12, 12, 64}}) {
      SCOPED_TRACE(std::string(dtype_name(cache_dtype)) + " h" + std::to_string(s.n_heads) + " kv" +
                   std::to_string(s.n_kv_heads) + " d" + std::to_string(s.head_dim));
      const size_t mark = arena.mark();
      const int width = (s.n_heads + 2 * s.n_kv_heads) * s.head_dim;
      Tensor k = arena.allocate(cache_dtype, {max_seq, s.n_kv_heads, s.head_dim});
      Tensor v = arena.allocate(cache_dtype, {max_seq, s.n_kv_heads, s.head_dim});
      Tensor q_all = arena.allocate(DType::F16, {1024, s.n_heads, s.head_dim});
      // Fill the cache 1024 rows at a time through split_qkv (one random block, reused).
      Tensor qkv_block = to_device(arena, random_f16_values(size_t(1024) * width, 1, -2.f, 2.f),
                                   DType::F16, {1024, width});
      for (int r0 = 0; r0 < max_seq; r0 += 1024) {
        const int n = std::min(1024, max_seq - r0);
        std::vector<int32_t> pos(n);
        for (int i = 0; i < n; ++i) pos[i] = r0 + i;
        const size_t m2 = arena.mark();
        Tensor qkv = qkv_block.slice_rows(0, n);
        Tensor q_view = q_all.slice_rows(0, n);
        kernels::split_qkv(qkv, Tensor(), s.n_heads, s.n_kv_heads, q_view, k, v,
                           to_device_i32(arena, pos), Tensor(), nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        arena.rewind(m2);
      }
      Tensor scratch = arena.allocate(
          DType::F32, {int64_t(kernels::attention_decode_scratch_floats(s.n_heads, s.head_dim, max_seq))});
      const float scale = 1.0f / std::sqrt(float(s.head_dim));
      for (int ctx : {1, 160, 161, 5000, max_seq}) {
        Tensor q = to_device(arena, random_f16_values(size_t(s.n_heads) * s.head_dim, ctx, -2.f, 2.f),
                             DType::F16, {1, s.n_heads, s.head_dim});
        Tensor d_pos = to_device_i32(arena, {ctx - 1});
        Tensor split = arena.allocate(DType::F16, {1, s.n_heads, s.head_dim});
        Tensor naive = arena.allocate(DType::F16, {1, s.n_heads, s.head_dim});
        kernels::attention_decode(q, k, v, d_pos, scale, split, scratch, nullptr);
        kernels::detail::attention_naive(q, k, v, d_pos, scale, naive, nullptr, "test");
        EXPECT_LT(max_abs_diff(to_host(split), to_host(naive)), 2e-3f) << "ctx=" << ctx;
      }
      arena.rewind(mark);
    }
  }
}
