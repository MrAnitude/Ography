// FlashAttention-2 style causal attention over the KV cache, on tensor cores (WMMA).
//
// One block per (query tile of 64 rows, head); 4 warps, each owning 16 query rows. K/V tiles
// of Bc keys are staged in shared memory (converted from fp8 if the cache is fp8) and shared by
// all warps. Per tile and warp:
//   S = Q K^T     WMMA 16x16x16, fp32 accumulate, stored to shared memory
//   online softmax on S (each lane: half a row), P written back as fp16 over S
//   O += P V      WMMA, accumulated into registers (lane: 8 of every 16 columns of its row)
// WMMA's fragment layout is opaque, which is why S and P V go through shared memory; the
// softmax state (m, l) and O never leave registers. Rows are masked by their own positions,
// so chunks at any offset (prefill after prefix reuse) work, as with the naive kernel.

#include <mma.h>

#include "core/cuda_check.h"
#include "kernels/attention.cuh"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels::detail {

namespace {

using namespace nvcuda;

constexpr int kBr = 64;     // query rows per block
constexpr int kWarps = 4;   // 16 rows each
constexpr int kPvLd = 20;   // leading dim of the per-warp 16x16 P V staging tile (floats)

template <int D, int Bc>
struct FlashSmem {
  static constexpr int kLdK = D + 8;    // halves; padding avoids bank conflicts
  static constexpr int kLdS = Bc + 4;   // floats
  static constexpr int kLdP = Bc + 8;   // halves (P aliases S)
  static constexpr size_t kKv = 2 * size_t(Bc) * kLdK * sizeof(__half);
  static constexpr size_t kS = size_t(kWarps) * 16 * kLdS * sizeof(float);
  static constexpr size_t kPv = size_t(kWarps) * 16 * kPvLd * sizeof(float);
  static constexpr size_t kBytes = kKv + kS + kPv;
  static_assert(2 * Bc >= kBr, "Q is staged in the K/V area");
};

// Copies rows [j0, j0 + Bc) of one KV head into shared memory as fp16 (zeros past `last`).
template <int D, int Bc, typename CacheT>
__device__ void load_kv_tile(__half* dst, const CacheT* __restrict__ cache, int j0, int last,
                             size_t kv_row, int kv_h) {
  constexpr int kLd = FlashSmem<D, Bc>::kLdK;
  for (int idx = threadIdx.x; idx < Bc * D / 8; idx += blockDim.x) {
    const int row = idx / (D / 8);
    const int c = (idx % (D / 8)) * 8;
    const int j = j0 + row;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (j <= last) {
      const CacheT* src = cache + j * kv_row + size_t(kv_h) * D + c;
      if constexpr (sizeof(CacheT) == 2) {
        v = *reinterpret_cast<const uint4*>(src);
      } else {
        __half h[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) h[i] = __float2half(to_float(src[i]));
        v = *reinterpret_cast<const uint4*>(h);
      }
    }
    *reinterpret_cast<uint4*>(dst + row * kLd + c) = v;
  }
}

template <int D, int Bc, typename CacheT>
__global__ void __launch_bounds__(kWarps * 32)
    flash_attention_kernel(const __half* __restrict__ q, const CacheT* __restrict__ k_cache,
                           const CacheT* __restrict__ v_cache,
                           const int32_t* __restrict__ positions, __half* __restrict__ out,
                           int T, int n_heads, int n_kv_heads, int max_seq, float scale_log2) {
  using S = FlashSmem<D, Bc>;
  extern __shared__ __align__(128) unsigned char smem[];
  __half* Ks = reinterpret_cast<__half*>(smem);
  __half* Vs = Ks + Bc * S::kLdK;
  float* Sw = reinterpret_cast<float*>(smem + S::kKv) + (threadIdx.x / 32) * 16 * S::kLdS;
  __half* Pw = reinterpret_cast<__half*>(Sw);  // P overwrites S, row by row
  float* PVw = reinterpret_cast<float*>(smem + S::kKv + S::kS) + (threadIdx.x / 32) * 16 * kPvLd;
  __shared__ int s_maxpos;

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int h = blockIdx.y;
  const int kv_h = h / (n_heads / n_kv_heads);
  const int t0 = blockIdx.x * kBr;
  const size_t q_row = size_t(n_heads) * D;
  const size_t kv_row = size_t(n_kv_heads) * D;

  // Furthest key any row of this block attends to.
  if (threadIdx.x == 0) s_maxpos = -1;
  __syncthreads();
  if (threadIdx.x < kBr && t0 + threadIdx.x < T) atomicMax(&s_maxpos, positions[t0 + threadIdx.x]);

  // Stage Q (zero rows past T) in the K/V area, then keep it in fragments.
  for (int idx = threadIdx.x; idx < kBr * D / 8; idx += blockDim.x) {
    const int row = idx / (D / 8);
    const int c = (idx % (D / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (t0 + row < T) v = *reinterpret_cast<const uint4*>(q + (t0 + row) * q_row + size_t(h) * D + c);
    *reinterpret_cast<uint4*>(Ks + row * S::kLdK + c) = v;
  }
  __syncthreads();
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf[D / 16];
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk) {
    wmma::load_matrix_sync(qf[kk], Ks + warp * 16 * S::kLdK + kk * 16, S::kLdK);
  }
  const int last_key = min(s_maxpos, max_seq - 1);

  // This lane's share: row r of the warp's 16, half hc of each 16-column block.
  const int r = lane / 2;
  const int hc = lane % 2;
  const int t = t0 + warp * 16 + r;
  const int my_pos = t < T ? positions[t] : -1;
  float m = -INFINITY, l = 0.f;
  float o[D / 2] = {};

  for (int j0 = 0; j0 <= last_key; j0 += Bc) {
    __syncthreads();  // everyone is done with the previous tile (and with Q staging)
    load_kv_tile<D, Bc>(Ks, k_cache, j0, last_key, kv_row, kv_h);
    load_kv_tile<D, Bc>(Vs, v_cache, j0, last_key, kv_row, kv_h);
    __syncthreads();

    // S = Q K^T (K row-major [Bc, D] read as column-major [D, Bc] = K^T).
#pragma unroll
    for (int n = 0; n < Bc / 16; ++n) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kb;
        wmma::load_matrix_sync(kb, Ks + n * 16 * S::kLdK + kk * 16, S::kLdK);
        wmma::mma_sync(acc, qf[kk], kb, acc);
      }
      wmma::store_matrix_sync(Sw + n * 16, acc, S::kLdS, wmma::mem_row_major);
    }
    __syncwarp();

    // Online softmax over this lane's half row (base-2 exponent, scale folded in).
    constexpr int kHalf = Bc / 2;
    float p[kHalf];
    float row_max = -INFINITY;
#pragma unroll
    for (int i = 0; i < kHalf; ++i) {
      const int j = j0 + hc * kHalf + i;
      p[i] = j <= my_pos ? Sw[r * S::kLdS + hc * kHalf + i] * scale_log2 : -INFINITY;
      row_max = fmaxf(row_max, p[i]);
    }
    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, 1));
    const float m_new = fmaxf(m, row_max);
    float corr = 1.f, row_sum = 0.f;
    if (m_new != -INFINITY) {
      corr = m == -INFINITY ? 0.f : exp2f(m - m_new);
#pragma unroll
      for (int i = 0; i < kHalf; ++i) {
        p[i] = exp2f(p[i] - m_new);  // exp2(-inf) = 0 for masked keys
        row_sum += p[i];
      }
    } else {
#pragma unroll
      for (int i = 0; i < kHalf; ++i) p[i] = 0.f;
    }
    row_sum += __shfl_xor_sync(0xffffffff, row_sum, 1);
    l = l * corr + row_sum;
    m = m_new;
#pragma unroll
    for (int i = 0; i < D / 2; ++i) o[i] *= corr;
    __syncwarp();  // all S reads done before P overwrites it
#pragma unroll
    for (int i = 0; i < kHalf; ++i) Pw[r * S::kLdP + hc * kHalf + i] = __float2half(p[i]);
    __syncwarp();

    // O += P V, one 16-column block at a time through a small staging tile.
#pragma unroll
    for (int n = 0; n < D / 16; ++n) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
      wmma::fill_fragment(acc, 0.f);
#pragma unroll
      for (int kk = 0; kk < Bc / 16; ++kk) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb;
        wmma::load_matrix_sync(pa, Pw + kk * 16, S::kLdP);
        wmma::load_matrix_sync(vb, Vs + kk * 16 * S::kLdK + n * 16, S::kLdK);
        wmma::mma_sync(acc, pa, vb, acc);
      }
      wmma::store_matrix_sync(PVw, acc, kPvLd, wmma::mem_row_major);
      __syncwarp();
#pragma unroll
      for (int i = 0; i < 8; ++i) o[n * 8 + i] += PVw[r * kPvLd + hc * 8 + i];
      __syncwarp();
    }
  }

  if (my_pos < 0) return;
  const float inv_l = 1.f / l;
  __half* o_row = out + t * q_row + size_t(h) * D;
#pragma unroll
  for (int n = 0; n < D / 16; ++n) {
#pragma unroll
    for (int i = 0; i < 8; ++i) o_row[n * 16 + hc * 8 + i] = __float2half(o[n * 8 + i] * inv_l);
  }
}

template <int D, int Bc, typename CacheT>
void launch_flash(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                  const Tensor& positions, float scale, Tensor& out, cudaStream_t stream) {
  const int T = static_cast<int>(q.dim(0));
  const int n_heads = static_cast<int>(q.dim(1));
  const int n_kv_heads = static_cast<int>(k_cache.dim(1));
  constexpr size_t smem = FlashSmem<D, Bc>::kBytes;
  auto kernel = flash_attention_kernel<D, Bc, CacheT>;
  static bool configured = false;  // per instantiation; attribute set once, before any capture
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(smem)));
    configured = true;
  }
  const dim3 grid((T + kBr - 1) / kBr, n_heads);
  kernel<<<grid, kWarps * 32, smem, stream>>>(
      q.data_as<const __half>(), k_cache.data_as<const CacheT>(),
      v_cache.data_as<const CacheT>(), positions.data_as<const int32_t>(), out.data_as<__half>(),
      T, n_heads, n_kv_heads, static_cast<int>(k_cache.dim(0)), scale * 1.4426950408889634f);
  CUDA_CHECK_LAUNCH();
}

}  // namespace

bool attention_flash(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                     const Tensor& positions, float scale, Tensor& out, cudaStream_t stream) {
  const int head_dim = static_cast<int>(q.dim(2));
  const bool fp8 = k_cache.dtype() == DType::FP8_E4M3;
  if (head_dim == 128) {
    fp8 ? launch_flash<128, 32, __nv_fp8_e4m3>(q, k_cache, v_cache, positions, scale, out, stream)
        : launch_flash<128, 32, __half>(q, k_cache, v_cache, positions, scale, out, stream);
    return true;
  }
  if (head_dim == 64) {
    fp8 ? launch_flash<64, 32, __nv_fp8_e4m3>(q, k_cache, v_cache, positions, scale, out, stream)
        : launch_flash<64, 32, __half>(q, k_cache, v_cache, positions, scale, out, stream);
    return true;
  }
  return false;  // other head sizes use the naive kernel
}

}  // namespace ography::kernels::detail
