// Decode attention (1 query token): split-K "flash-decoding" with GQA grouping.
//
// Grid (kv_head, split): each block covers split_len keys of one KV head for all n_heads /
// n_kv_heads query heads that share it, so K/V are read from memory once per group rather than
// once per query head, and long contexts spread over many blocks instead of one per head.
// Within a block, each warp takes 32-key chunks: lane j scores key j against every head (K
// chunk staged in padded shared memory), one warp reduction per head per chunk for the softmax,
// then V rows (read coalesced) are accumulated into per-lane slices of each head's output.
// Blocks write partial softmax states (m, l, unnormalized O); a second kernel merges them.
//
// split_len depends only on the cache capacity (max_seq), never on the current length, and the
// grid covers the whole capacity: blocks past the current length exit at once. So the launch is
// identical every step and can be replayed from a CUDA graph.

#include "core/cuda_check.h"
#include "kernels/attention.cuh"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

namespace {

using namespace detail;

constexpr int kWarps = 4;
constexpr int kChunk = 32;     // keys per warp iteration (one per lane)
constexpr int kMaxGroup = 8;   // query heads per KV head handled by the fast path
constexpr int kMaxSplits = 256;
constexpr int kMinSplitLen = 128;

int split_len_for(int max_seq) {
  const int len = (max_seq + kMaxSplits - 1) / kMaxSplits;
  return std::max(kMinSplitLen, (len + kChunk - 1) / kChunk * kChunk);
}

template <int D, typename CacheT>
__global__ void __launch_bounds__(kWarps * 32)
    decode_split_kernel(const __half* __restrict__ q, const CacheT* __restrict__ k_cache,
                        const CacheT* __restrict__ v_cache, const int32_t* __restrict__ positions,
                        float* __restrict__ part_o, float* __restrict__ part_ml, int n_heads,
                        int n_kv_heads, int max_seq, int split_len, int max_splits,
                        float scale_log2) {
  constexpr int kPer = D / 32;          // output dims per lane
  constexpr int kLdK = D / 2 + 1;       // K row stride in half2 words (odd: no bank conflicts)
  // The K staging area is reused for the warps' partial outputs once the key loop is done.
  constexpr size_t kKBytes = sizeof(__half2) * kWarps * kChunk * kLdK;
  constexpr size_t kOBytes = sizeof(float) * kWarps * kMaxGroup * D;
  static_assert(kOBytes <= kKBytes, "s_o must fit in the K staging area");
  __shared__ __align__(16) unsigned char s_raw[kKBytes];
  auto s_k = reinterpret_cast<__half2(*)[kChunk * kLdK]>(s_raw);
  auto s_o = reinterpret_cast<float(*)[kMaxGroup][D]>(s_raw);
  __shared__ float s_q[kMaxGroup][D];
  __shared__ float s_p[kWarps][kMaxGroup][kChunk];
  __shared__ float s_m[kWarps][kMaxGroup], s_l[kWarps][kMaxGroup];

  const int kv_h = blockIdx.x;
  const int split = blockIdx.y;
  const int group = n_heads / n_kv_heads;
  const int n_keys = min(positions[0] + 1, max_seq);
  const int begin = split * split_len;
  if (begin >= n_keys) return;  // past the current length (uniform across the block)
  const int end = min(n_keys, begin + split_len);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const size_t kv_row = size_t(n_kv_heads) * D;
  const int h0 = kv_h * group;

  for (int i = threadIdx.x; i < group * D; i += blockDim.x) {
    s_q[i / D][i % D] = __half2float(q[size_t(h0 + i / D) * D + i % D]) * scale_log2;
  }
  __syncthreads();

  float m[kMaxGroup], l[kMaxGroup], acc[kMaxGroup][kPer];
#pragma unroll
  for (int g = 0; g < kMaxGroup; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.f;
#pragma unroll
    for (int i = 0; i < kPer; ++i) acc[g][i] = 0.f;
  }

  for (int c0 = begin + warp * kChunk; c0 < end; c0 += kWarps * kChunk) {
    const int n = min(kChunk, end - c0);
    // Stage the K chunk: each lane copies 2 dims (one half2) of every row, coalesced.
    for (int j = 0; j < n; ++j) {
      const CacheT* row = k_cache + size_t(c0 + j) * kv_row + size_t(kv_h) * D;
#pragma unroll
      for (int d2 = lane; d2 < D / 2; d2 += 32) {
        s_k[warp][j * kLdK + d2] = __floats2half2_rn(to_float(row[2 * d2]), to_float(row[2 * d2 + 1]));
      }
    }
    __syncwarp();

    // Lane j: scores of key c0 + j for every head of the group.
    float s[kMaxGroup];
#pragma unroll
    for (int g = 0; g < kMaxGroup; ++g) s[g] = 0.f;
    if (lane < n) {
      for (int d2 = 0; d2 < D / 2; ++d2) {
        const float2 kv = __half22float2(s_k[warp][lane * kLdK + d2]);
#pragma unroll
        for (int g = 0; g < kMaxGroup; ++g) {
          if (g < group) s[g] += s_q[g][2 * d2] * kv.x + s_q[g][2 * d2 + 1] * kv.y;
        }
      }
    }
    // Online softmax per head over this chunk.
#pragma unroll
    for (int g = 0; g < kMaxGroup; ++g) {
      if (g >= group) break;
      float mx = lane < n ? s[g] : -INFINITY;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
      const float m_new = fmaxf(m[g], mx);
      const float corr = exp2f(m[g] - m_new);  // m = -inf on the first chunk -> 0
      const float p = lane < n ? exp2f(s[g] - m_new) : 0.f;
      l[g] = l[g] * corr + warp_sum(p);
      m[g] = m_new;
#pragma unroll
      for (int i = 0; i < kPer; ++i) acc[g][i] *= corr;
      s_p[warp][g][lane] = p;
    }
    __syncwarp();
    // acc[g][dims of this lane] += sum_j p[g][j] * V[j][dims].
    for (int j = 0; j < n; ++j) {
      const CacheT* row = v_cache + size_t(c0 + j) * kv_row + size_t(kv_h) * D;
      float v[kPer];
#pragma unroll
      for (int i = 0; i < kPer; ++i) v[i] = to_float(row[lane + 32 * i]);
#pragma unroll
      for (int g = 0; g < kMaxGroup; ++g) {
        if (g >= group) break;
        const float p = s_p[warp][g][j];
#pragma unroll
        for (int i = 0; i < kPer; ++i) acc[g][i] += p * v[i];
      }
    }
    __syncwarp();
  }

  // Merge the warps' states, then write this split's partial result.
  __syncthreads();  // every warp is done with s_k before s_o overwrites it
  for (int g = 0; g < group; ++g) {
    if (lane == 0) {
      s_m[warp][g] = m[g];
      s_l[warp][g] = l[g];
    }
#pragma unroll
    for (int i = 0; i < kPer; ++i) s_o[warp][g][lane + 32 * i] = acc[g][i];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < group * D; i += blockDim.x) {
    const int g = i / D, d = i % D;
    float mx = -INFINITY;
    for (int w = 0; w < kWarps; ++w) mx = fmaxf(mx, s_m[w][g]);
    float o = 0.f, lsum = 0.f;
    for (int w = 0; w < kWarps; ++w) {
      const float f = s_m[w][g] == -INFINITY ? 0.f : exp2f(s_m[w][g] - mx);
      o += s_o[w][g][d] * f;
      lsum += s_l[w][g] * f;
    }
    const size_t slot = size_t(h0 + g) * max_splits + split;
    part_o[slot * D + d] = o;
    if (d == 0) {
      part_ml[2 * slot] = mx;
      part_ml[2 * slot + 1] = lsum;
    }
  }
}

// One block per head: merges the splits that hold keys.
template <int D>
__global__ void decode_combine_kernel(const float* __restrict__ part_o,
                                      const float* __restrict__ part_ml,
                                      const int32_t* __restrict__ positions,
                                      __half* __restrict__ out, int max_seq, int split_len,
                                      int max_splits) {
  const int h = blockIdx.x;
  const int n_keys = min(positions[0] + 1, max_seq);
  const int splits = (n_keys + split_len - 1) / split_len;
  float mx = -INFINITY;
  for (int s = 0; s < splits; ++s) mx = fmaxf(mx, part_ml[2 * (size_t(h) * max_splits + s)]);
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float o = 0.f, l = 0.f;
    for (int s = 0; s < splits; ++s) {
      const size_t slot = size_t(h) * max_splits + s;
      const float f = exp2f(part_ml[2 * slot] - mx);
      o += part_o[slot * D + d] * f;
      l += part_ml[2 * slot + 1] * f;
    }
    out[size_t(h) * D + d] = __float2half(o / l);
  }
}

template <int D, typename CacheT>
void launch_decode(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                   const Tensor& positions, float scale, Tensor& out, Tensor& scratch,
                   cudaStream_t stream) {
  const int n_heads = static_cast<int>(q.dim(1));
  const int n_kv_heads = static_cast<int>(k_cache.dim(1));
  const int max_seq = static_cast<int>(k_cache.dim(0));
  const int split_len = split_len_for(max_seq);
  const int max_splits = (max_seq + split_len - 1) / split_len;
  float* part_o = scratch.data_as<float>();
  float* part_ml = part_o + size_t(n_heads) * max_splits * D;
  decode_split_kernel<D, CacheT><<<dim3(n_kv_heads, max_splits), kWarps * 32, 0, stream>>>(
      q.data_as<const __half>(), k_cache.data_as<const CacheT>(), v_cache.data_as<const CacheT>(),
      positions.data_as<const int32_t>(), part_o, part_ml, n_heads, n_kv_heads, max_seq,
      split_len, max_splits, scale * 1.4426950408889634f);
  CUDA_CHECK_LAUNCH();
  decode_combine_kernel<D><<<n_heads, D, 0, stream>>>(part_o, part_ml,
                                                      positions.data_as<const int32_t>(),
                                                      out.data_as<__half>(), max_seq, split_len,
                                                      max_splits);
  CUDA_CHECK_LAUNCH();
}

}  // namespace

size_t attention_decode_scratch_floats(int n_heads, int head_dim, int max_seq) {
  const int split_len = split_len_for(max_seq);
  const size_t splits = (max_seq + split_len - 1) / split_len;
  return size_t(n_heads) * splits * (head_dim + 2);
}

void attention_decode(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                      const Tensor& positions, float scale, Tensor& out, Tensor& scratch,
                      cudaStream_t stream) {
  const char* fn = "attention_decode";
  detail::check(q.ndim() == 3 && q.dim(0) == 1, fn, "q must be [1, heads, dim]");
  detail::check_attention(q, k_cache, v_cache, positions, out, fn);
  const int n_heads = static_cast<int>(q.dim(1));
  const int head_dim = static_cast<int>(q.dim(2));
  const int group = n_heads / static_cast<int>(k_cache.dim(1));
  const bool fast =
      (head_dim == 64 || head_dim == 128) && group <= kMaxGroup && !scratch.empty();
  if (!fast) {
    detail::attention_naive(q, k_cache, v_cache, positions, scale, out, stream, fn);
    return;
  }
  detail::check_cuda(scratch, DType::F32, fn, "scratch");
  detail::check(static_cast<size_t>(scratch.numel()) >=
                    attention_decode_scratch_floats(n_heads, head_dim,
                                                    static_cast<int>(k_cache.dim(0))),
                fn, "scratch smaller than attention_decode_scratch_floats()");
  const bool fp8 = k_cache.dtype() == DType::FP8_E4M3;
  if (head_dim == 128) {
    fp8 ? launch_decode<128, __nv_fp8_e4m3>(q, k_cache, v_cache, positions, scale, out, scratch, stream)
        : launch_decode<128, __half>(q, k_cache, v_cache, positions, scale, out, scratch, stream);
  } else {
    fp8 ? launch_decode<64, __nv_fp8_e4m3>(q, k_cache, v_cache, positions, scale, out, scratch, stream)
        : launch_decode<64, __half>(q, k_cache, v_cache, positions, scale, out, scratch, stream);
  }
}

}  // namespace ography::kernels
