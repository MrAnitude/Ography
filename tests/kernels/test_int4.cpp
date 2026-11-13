// INT4 weights: AWQ / GPTQ repacking, dequantization, and the W4A16 GEMM (fused GEMV for T=1,
// tiled dequant + cuBLAS for T>1) vs CPU references.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include <cublas_v2.h>

#include "kernels/kernels.h"
#include "test_util.h"

using namespace ography;
using namespace ography::test;
using kernels::Int4Source;

namespace {

// A random group-quantized weight W [out, in] (q in 0..15, per-(row, group) zero and scale).
struct Int4Weight {
  int out, in, group;
  std::vector<uint8_t> q;      // [out, in]
  std::vector<uint8_t> zero;   // [out, groups]
  std::vector<float> scale;    // [out, groups], fp16-representable
  int groups() const { return in / group; }
  float w(int o, int i) const {
    const int g = o * groups() + i / group;
    return (float(q[o * in + i]) - float(zero[g])) * scale[g];
  }
};

Int4Weight random_weight(int out, int in, int group, uint32_t seed) {
  std::mt19937 rng(seed);
  Int4Weight w{out, in, group, {}, {}, {}};
  w.q.resize(size_t(out) * in);
  for (auto& v : w.q) v = rng() % 16;
  w.zero.resize(size_t(out) * w.groups());
  for (auto& z : w.zero) z = 1 + rng() % 15;  // GPTQ can't store zero 0 (it stores zero - 1)
  w.scale = random_f16_values(size_t(out) * w.groups(), seed + 1, 0.001f, 0.02f);
  return w;
}

// Checkpoint layouts, packed on the CPU straight from the format descriptions.
const int kAwqNibble[8] = {0, 4, 1, 5, 2, 6, 3, 7};  // column 8b+j -> nibble kAwqNibble[j]

struct Packed {
  std::vector<int32_t> qweight, qzeros;
  std::vector<float> scales;  // [groups, out]
  std::vector<int64_t> qweight_shape;
};

Packed pack(const Int4Weight& w, Int4Source src) {
  Packed p;
  const int G = w.groups();
  auto put = [](std::vector<int32_t>& v, size_t idx, int nib, uint32_t val) {
    v[idx] = int32_t(uint32_t(v[idx]) | (val << (4 * nib)));
  };
  if (src == Int4Source::AWQ) {
    p.qweight.assign(size_t(w.in) * (w.out / 8), 0);
    p.qweight_shape = {w.in, w.out / 8};
    for (int i = 0; i < w.in; ++i)
      for (int o = 0; o < w.out; ++o)
        put(p.qweight, size_t(i) * (w.out / 8) + o / 8, kAwqNibble[o % 8], w.q[o * w.in + i]);
  } else {
    p.qweight.assign(size_t(w.in / 8) * w.out, 0);
    p.qweight_shape = {w.in / 8, w.out};
    for (int i = 0; i < w.in; ++i)
      for (int o = 0; o < w.out; ++o)
        put(p.qweight, size_t(i / 8) * w.out + o, i % 8, w.q[o * w.in + i]);
  }
  p.qzeros.assign(size_t(G) * (w.out / 8), 0);
  p.scales.resize(size_t(G) * w.out);
  for (int g = 0; g < G; ++g)
    for (int o = 0; o < w.out; ++o) {
      const uint32_t z = w.zero[o * G + g];
      if (src == Int4Source::AWQ) put(p.qzeros, size_t(g) * (w.out / 8) + o / 8, kAwqNibble[o % 8], z);
      else put(p.qzeros, size_t(g) * (w.out / 8) + o / 8, o % 8, z - 1);
      p.scales[size_t(g) * w.out + o] = w.scale[o * G + g];
    }
  return p;
}

Tensor upload_i32(DeviceArena& arena, const std::vector<int32_t>& v, std::vector<int64_t> shape) {
  Tensor t = arena.allocate(DType::I32, std::move(shape));
  CUDA_CHECK(cudaMemcpy(t.data(), v.data(), t.nbytes(), cudaMemcpyHostToDevice));
  return t;
}

struct EngineInt4 {
  Tensor qweight, scales, zeros;
};

EngineInt4 repack_on_gpu(DeviceArena& arena, const Int4Weight& w, Int4Source src) {
  const Packed p = pack(w, src);
  EngineInt4 e{arena.allocate(DType::I32, {w.out, w.in / 8}),
               arena.allocate(DType::F16, {w.out, w.groups()}),
               arena.allocate(DType::F16, {w.out, w.groups()})};
  // Repack in two halves of rows to exercise row_offset (as for fused projections).
  const int half = w.out / 16 * 8;  // checkpoint packing needs multiples of 8 rows
  for (int part = 0; part < 2; ++part) {
    Int4Weight sub = w;
    const int r0 = part * half, rows = part ? w.out - half : half;
    sub.out = rows;
    sub.q.assign(w.q.begin() + size_t(r0) * w.in, w.q.begin() + size_t(r0 + rows) * w.in);
    sub.zero.assign(w.zero.begin() + size_t(r0) * w.groups(), w.zero.begin() + size_t(r0 + rows) * w.groups());
    sub.scale.assign(w.scale.begin() + size_t(r0) * w.groups(), w.scale.begin() + size_t(r0 + rows) * w.groups());
    const Packed sp = pack(sub, src);
    Tensor qw = upload_i32(arena, sp.qweight, sp.qweight_shape);
    Tensor qz = upload_i32(arena, sp.qzeros, {sub.groups(), rows / 8});
    Tensor sc = to_device(arena, sp.scales, DType::F16, {sub.groups(), rows});
    kernels::repack_int4(src, qw, qz, sc, w.group, e.qweight, e.scales, e.zeros, r0, nullptr);
  }
  (void)p;
  return e;
}

}  // namespace

TEST(Int4, RepackAndDequantMatchFormat) {
  DeviceArena arena(size_t{64} << 20);
  for (Int4Source src : {Int4Source::AWQ, Int4Source::GPTQ}) {
    SCOPED_TRACE(src == Int4Source::AWQ ? "AWQ" : "GPTQ");
    const Int4Weight w = random_weight(/*out=*/96, /*in=*/512, /*group=*/128, 7);
    const EngineInt4 e = repack_on_gpu(arena, w, src);
    Tensor deq = arena.allocate(DType::F16, {w.out, w.in});
    kernels::dequant_int4(e.qweight, e.scales, e.zeros, w.group, 0, deq, nullptr);
    const auto got = to_host(deq);
    float max_err = 0;
    for (int o = 0; o < w.out; ++o)
      for (int i = 0; i < w.in; ++i) max_err = std::max(max_err, std::fabs(got[o * w.in + i] - w.w(o, i)));
    EXPECT_LT(max_err, 2e-4f);  // fp16 rounding of values up to ~0.3
  }
}

TEST(Int4, GemmMatchesCpu) {
  DeviceArena arena(size_t{256} << 20);
  cublasHandle_t cublas;
  CUBLAS_CHECK(cublasCreate(&cublas));
  struct Shape { int out, in, group; };
  for (const Shape s : {Shape{256, 1536, 128}, Shape{1536, 8960, 128}, Shape{3584, 18944, 128},
                        Shape{72, 256, 64}}) {
    SCOPED_TRACE(std::to_string(s.out) + "x" + std::to_string(s.in));
    const size_t mark = arena.mark();
    const Int4Weight w = random_weight(s.out, s.in, s.group, s.out + s.in);
    const EngineInt4 e = repack_on_gpu(arena, w, Int4Source::AWQ);
    for (int T : {1, 3, 8, 13}) {
      const auto x = random_f16_values(size_t(T) * s.in, T, -1.f, 1.f);
      std::vector<float> want(size_t(T) * s.out);
      double max_ref = 0;
      for (int t = 0; t < T; ++t)
        for (int o = 0; o < s.out; ++o) {
          double acc = 0;
          for (int i = 0; i < s.in; ++i) acc += double(w.w(o, i)) * x[size_t(t) * s.in + i];
          want[size_t(t) * s.out + o] = float(acc);
          max_ref = std::max(max_ref, std::fabs(acc));
        }
      Tensor d_x = to_device(arena, x, DType::F16, {T, s.in});
      Tensor out = arena.allocate(DType::F16, {T, s.out});
      // Room for ~3 tiles of rows, so the tiled path loops.
      Tensor scratch = arena.allocate(DType::F16, {int64_t(s.in) * (s.out / 3 + 1)});
      kernels::gemm_w4a16(cublas, d_x, e.qweight, e.scales, e.zeros, s.group, out, scratch, nullptr);
      const float err = max_abs_diff(to_host(out), want);
      EXPECT_LT(err, 2e-3f * float(max_ref) + 1e-3f) << "T=" << T << " max|ref|=" << max_ref;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    arena.rewind(mark);
  }
  cublasDestroy(cublas);
}
