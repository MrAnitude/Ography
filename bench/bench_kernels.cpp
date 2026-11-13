// Kernel microbenchmarks timed with cudaEvents (warm-up, then the median of N runs), on the
// shapes of the models we run. Reports time, and achieved bandwidth or TFLOPS where useful
// (the 4070 Laptop peaks at ~256 GB/s; decode kernels are bandwidth-bound).
//
//   ./bench_kernels [filter]      e.g. ./bench_kernels attention
// For deeper analysis: ncu --set full -k regex:attention ./bench_kernels attention

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "core/cuda_check.h"
#include "core/memory.h"
#include "kernels/kernels.h"

using namespace ography;

namespace {

const char* g_filter = nullptr;

// Median milliseconds of fn over `iters` runs after warm-up.
double time_ms(const std::function<void()>& fn, int iters = 20) {
  for (int i = 0; i < 3; ++i) fn();
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<float> ms(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(a));
    fn();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ms[i], a, b));
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  std::sort(ms.begin(), ms.end());
  return ms[iters / 2];
}

bool enabled(const char* name) { return !g_filter || std::strstr(name, g_filter); }

void fill_random(Tensor& t, uint32_t seed, float scale = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  if (t.dtype() == DType::F16) {
    std::vector<__half> h(t.numel());
    for (auto& v : h) v = __float2half(dist(rng));
    CUDA_CHECK(cudaMemcpy(t.data(), h.data(), t.nbytes(), cudaMemcpyHostToDevice));
  } else if (t.dtype() == DType::I32) {
    std::vector<int32_t> h(t.numel());
    for (auto& v : h) v = static_cast<int32_t>(rng());
    CUDA_CHECK(cudaMemcpy(t.data(), h.data(), t.nbytes(), cudaMemcpyHostToDevice));
  }
}

Tensor positions(DeviceArena& arena, int start, int n) {
  std::vector<int32_t> p(n);
  for (int i = 0; i < n; ++i) p[i] = start + i;
  Tensor t = arena.allocate(DType::I32, {n});
  CUDA_CHECK(cudaMemcpy(t.data(), p.data(), n * 4, cudaMemcpyHostToDevice));
  return t;
}

struct AttnShape {
  const char* model;
  int n_heads, n_kv_heads, head_dim;
};
const AttnShape kAttnShapes[] = {{"qwen2.5-7b", 28, 4, 128}, {"llama3.2-1b", 32, 8, 64}};

// Prefill of a whole prompt in 2048-token chunks (as the engine does), per layer.
void bench_attention_prefill(DeviceArena& arena) {
  if (!enabled("attention_prefill")) return;
  for (const auto& s : kAttnShapes) {
    for (int prompt : {512, 2048, 6144}) {
      const size_t mark = arena.mark();
      const int chunk = std::min(prompt, 2048);
      Tensor q = arena.allocate(DType::F16, {chunk, s.n_heads, s.head_dim});
      Tensor out = arena.allocate(DType::F16, {chunk, s.n_heads, s.head_dim});
      Tensor k = arena.allocate(DType::F16, {prompt, s.n_kv_heads, s.head_dim});
      Tensor v = arena.allocate(DType::F16, {prompt, s.n_kv_heads, s.head_dim});
      fill_random(q, 1);
      fill_random(k, 2);
      fill_random(v, 3);
      std::vector<Tensor> pos;
      for (int off = 0; off < prompt; off += chunk) pos.push_back(positions(arena, off, chunk));
      const float scale = 1.0f / std::sqrt(float(s.head_dim));
      const double ms = time_ms([&] {
        for (const Tensor& p : pos) kernels::attention_prefill(q, k, v, p, scale, out, nullptr);
      }, 5);
      // Causal attention FLOPs: 2 matmuls of 2*T*T*D/2 each per head.
      const double flops = 2.0 * 2.0 * prompt * double(prompt) / 2 * s.head_dim * s.n_heads;
      std::printf("attention_prefill  %-12s T=%5d  %8.2f ms/layer  %6.2f TFLOPS\n", s.model, prompt,
                  ms, flops / ms / 1e9);
      arena.rewind(mark);
    }
  }
}

// One decode step's attention at a given context length, per layer.
void bench_attention_decode(DeviceArena& arena) {
  if (!enabled("attention_decode")) return;
  for (const auto& s : kAttnShapes) {
    for (int ctx : {512, 4096, 16384}) {
      const size_t mark = arena.mark();
      const int max_seq = 22848;  // the engine's 7B AWQ capacity: split sizes depend on it
      Tensor q = arena.allocate(DType::F16, {1, s.n_heads, s.head_dim});
      Tensor out = arena.allocate(DType::F16, {1, s.n_heads, s.head_dim});
      Tensor k = arena.allocate(DType::F16, {max_seq, s.n_kv_heads, s.head_dim});
      Tensor v = arena.allocate(DType::F16, {max_seq, s.n_kv_heads, s.head_dim});
      fill_random(q, 1);
      fill_random(k, 2);
      fill_random(v, 3);
      Tensor p = positions(arena, ctx - 1, 1);
      Tensor scratch = arena.allocate(
          DType::F32,
          {int64_t(kernels::attention_decode_scratch_floats(s.n_heads, s.head_dim, max_seq))});
      const float scale = 1.0f / std::sqrt(float(s.head_dim));
      const double ms =
          time_ms([&] { kernels::attention_decode(q, k, v, p, scale, out, scratch, nullptr); });
      const double bytes = 2.0 * ctx * s.n_kv_heads * s.head_dim * 2;
      std::printf("attention_decode   %-12s ctx=%5d  %8.3f ms/layer  %6.1f GB/s\n", s.model, ctx, ms,
                  bytes / ms / 1e6);
      arena.rewind(mark);
    }
  }
}

// INT4 linears of Qwen2.5 7B at decode and short-prefill token counts.
void bench_w4a16(DeviceArena& arena, cublasHandle_t cublas) {
  if (!enabled("w4a16")) return;
  struct Lin { const char* name; int out, in; };
  const Lin lins[] = {{"qkv", 4608, 3584}, {"gate_up", 37888, 3584}, {"down", 3584, 18944}};
  Tensor scratch = arena.allocate(DType::F16, {int64_t{4} << 20});
  for (int T : {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 256}) {
    double total = 0, bytes = 0;
    for (const auto& l : lins) {
      const size_t mark = arena.mark();
      const int group = 128;
      Tensor qw = arena.allocate(DType::I32, {l.out, l.in / 8});
      Tensor sc = arena.allocate(DType::F16, {l.out, l.in / group});
      Tensor z = arena.allocate(DType::F16, {l.out, l.in / group});
      Tensor x = arena.allocate(DType::F16, {T, l.in});
      Tensor y = arena.allocate(DType::F16, {T, l.out});
      fill_random(qw, 1);
      fill_random(sc, 2, 0.01f);
      fill_random(z, 3);
      fill_random(x, 4);
      total += time_ms([&] { kernels::gemm_w4a16(cublas, x, qw, sc, z, group, y, scratch, nullptr); });
      bytes += double(qw.nbytes());
      arena.rewind(mark);
    }
    // x28 layers: the whole model's INT4 linears (lm_head excluded).
    std::printf("w4a16 qwen2.5-7b   T=%4d  %8.3f ms/layer  %7.2f ms/model  %6.1f GB/s (weights)\n",
                T, total, total * 28, bytes / total / 1e6);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) g_filter = argv[1];
  DeviceArena arena(size_t{1536} << 20);
  cublasHandle_t cublas;
  CUBLAS_CHECK(cublasCreate(&cublas));
  bench_attention_prefill(arena);
  bench_attention_decode(arena);
  bench_w4a16(arena, cublas);
  cublasDestroy(cublas);
  return 0;
}
