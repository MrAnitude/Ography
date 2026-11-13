// Naive causal attention over the KV cache, shared by prefill and decode for now.
//
// One block per (head, query token). The block's warps split the keys between them; within
// a warp each lane owns head_dim/32 dimensions. Each warp keeps an online-softmax state
// (running max m, normaliser l, weighted V sum acc) over its keys, and the warps' states are
// merged at the end. Memory use doesn't grow with sequence length, so this works for any
// context size. Everything is accumulated in fp32.
//
// TODO: FlashAttention-2 style tiled prefill (roadmap step 10).

#include <cmath>

#include "core/cuda_check.h"
#include "kernels/attention.cuh"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

namespace detail {

namespace {

constexpr int kWarps = 4;

template <int kPerLane>  // head_dim / 32, rounded up
__global__ void attention_naive_kernel(const __half* __restrict__ q,
                                       const __half* __restrict__ k_cache,
                                       const __half* __restrict__ v_cache,
                                       const int32_t* __restrict__ positions,
                                       __half* __restrict__ out, int n_heads, int n_kv_heads,
                                       int head_dim, int max_seq, float scale) {
  extern __shared__ float smem[];  // [kWarps] m, [kWarps] l, [kWarps][head_dim] acc
  float* s_m = smem;
  float* s_l = smem + kWarps;
  float* s_acc = smem + 2 * kWarps;

  const int h = blockIdx.x;
  const int t = blockIdx.y;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int kv_h = h / (n_heads / n_kv_heads);
  const int n_keys = min(positions[t] + 1, max_seq);
  const size_t kv_row = static_cast<size_t>(n_kv_heads) * head_dim;

  // Pre-scaled query slice owned by this lane.
  const __half* q_row = q + (static_cast<size_t>(t) * n_heads + h) * head_dim;
  float qv[kPerLane];
#pragma unroll
  for (int p = 0; p < kPerLane; ++p) {
    const int d = lane + 32 * p;
    qv[p] = d < head_dim ? __half2float(q_row[d]) * scale : 0.0f;
  }

  float m = -INFINITY;
  float l = 0.0f;
  float acc[kPerLane] = {};
  for (int key = warp; key < n_keys; key += kWarps) {
    const __half* k_row = k_cache + key * kv_row + static_cast<size_t>(kv_h) * head_dim;
    const __half* v_row = v_cache + key * kv_row + static_cast<size_t>(kv_h) * head_dim;
    float s = 0.0f;
#pragma unroll
    for (int p = 0; p < kPerLane; ++p) {
      const int d = lane + 32 * p;
      if (d < head_dim) s += qv[p] * __half2float(k_row[d]);
    }
    s = warp_sum(s);

    const float m_new = fmaxf(m, s);
    const float correction = m == -INFINITY ? 0.0f : expf(m - m_new);
    const float w = expf(s - m_new);
    l = l * correction + w;
#pragma unroll
    for (int p = 0; p < kPerLane; ++p) {
      const int d = lane + 32 * p;
      const float v = d < head_dim ? __half2float(v_row[d]) : 0.0f;
      acc[p] = acc[p] * correction + w * v;
    }
    m = m_new;
  }

  if (lane == 0) {
    s_m[warp] = m;
    s_l[warp] = l;
  }
#pragma unroll
  for (int p = 0; p < kPerLane; ++p) {
    const int d = lane + 32 * p;
    if (d < head_dim) s_acc[warp * head_dim + d] = acc[p];
  }
  __syncthreads();

  // Merge the warps' partial softmax states. Warp 0 always saw key 0, so m_max is finite.
  float m_max = -INFINITY;
  for (int w = 0; w < kWarps; ++w) m_max = fmaxf(m_max, s_m[w]);
  float l_total = 0.0f;
  float factor[kWarps];
  for (int w = 0; w < kWarps; ++w) {
    factor[w] = s_m[w] == -INFINITY ? 0.0f : expf(s_m[w] - m_max);
    l_total += s_l[w] * factor[w];
  }
  __half* o_row = out + (static_cast<size_t>(t) * n_heads + h) * head_dim;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float o = 0.0f;
    for (int w = 0; w < kWarps; ++w) o += s_acc[w * head_dim + d] * factor[w];
    o_row[d] = __float2half(o / l_total);
  }
}

}  // namespace

void attention_naive(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                     const Tensor& positions, float scale, Tensor& out, cudaStream_t stream,
                     const char* fn) {
  check_cuda(q, DType::F16, fn, "q");
  check_cuda(k_cache, DType::F16, fn, "k_cache");
  check_cuda(v_cache, DType::F16, fn, "v_cache");
  check_cuda(out, DType::F16, fn, "out");
  check_cuda(positions, DType::I32, fn, "positions");
  check(q.ndim() == 3 && k_cache.ndim() == 3, fn, "q must be [T, heads, dim], cache [seq, kv_heads, dim]");
  check(v_cache.shape() == k_cache.shape(), fn, "k_cache and v_cache must match");
  check(out.shape() == q.shape(), fn, "out must match q");
  const int T = static_cast<int>(q.dim(0));
  const int n_heads = static_cast<int>(q.dim(1));
  const int head_dim = static_cast<int>(q.dim(2));
  const int n_kv_heads = static_cast<int>(k_cache.dim(1));
  const int max_seq = static_cast<int>(k_cache.dim(0));
  check(k_cache.dim(2) == head_dim, fn, "head_dim mismatch between q and cache");
  check(n_kv_heads > 0 && n_heads % n_kv_heads == 0, fn, "n_heads must be a multiple of n_kv_heads");
  check(head_dim % 32 == 0 && head_dim <= 256, fn, "head_dim must be a multiple of 32, <= 256");
  check(positions.ndim() == 1 && positions.dim(0) == T, fn, "positions must be [T]");
  if (T == 0) return;

  const dim3 grid(n_heads, T);
  const int threads = kWarps * 32;
  const size_t smem = (2 * kWarps + static_cast<size_t>(kWarps) * head_dim) * sizeof(float);
  auto args = [&](auto kernel) {
    kernel<<<grid, threads, smem, stream>>>(
        q.data_as<const __half>(), k_cache.data_as<const __half>(),
        v_cache.data_as<const __half>(), positions.data_as<const int32_t>(),
        out.data_as<__half>(), n_heads, n_kv_heads, head_dim, max_seq, scale);
  };
  if (head_dim <= 64) {
    args(attention_naive_kernel<2>);
  } else if (head_dim <= 128) {
    args(attention_naive_kernel<4>);
  } else {
    args(attention_naive_kernel<8>);
  }
  CUDA_CHECK_LAUNCH();
}

}  // namespace detail

void attention_prefill(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                       const Tensor& positions, float scale, Tensor& out, cudaStream_t stream) {
  detail::attention_naive(q, k_cache, v_cache, positions, scale, out, stream,
                          "attention_prefill");
}

}  // namespace ography::kernels
