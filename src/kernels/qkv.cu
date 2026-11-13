// Splits the fused QKV projection into q and the KV cache (adding the fused bias and applying
// rotary embeddings on the way), so K/V land in the cache without an extra round trip.

#include "core/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

namespace {

using namespace detail;

// cos/sin of pos * inv_freq. The angle is formed in fp32 exactly as HF does; the range
// reduction to [-pi, pi] is done in double so the fast-math __sincosf stays accurate even at
// positions of 100K+ (where angles reach ~1e5 rad).
__device__ __forceinline__ void rope_cos_sin(int pos, float inv_freq, float& c, float& s) {
  const float angle = static_cast<float>(pos) * inv_freq;
  const double two_pi = 6.283185307179586;
  const double k = rint(static_cast<double>(angle) / two_pi);
  __sincosf(static_cast<float>(static_cast<double>(angle) - k * two_pi), &s, &c);
}

// One block per token. First the q and k heads, a (i, i + d/2) pair per thread iteration so
// rope can rotate them together; then v. CacheT is the KV cache element type.
template <typename CacheT>
__global__ void split_qkv_kernel(const __half* __restrict__ qkv, const __half* __restrict__ bias,
                                 const int32_t* __restrict__ positions,
                                 const float* __restrict__ inv_freq, __half* __restrict__ q,
                                 CacheT* __restrict__ k_cache, CacheT* __restrict__ v_cache,
                                 int n_heads, int n_kv_heads, int head_dim, int max_seq) {
  const int t = blockIdx.x;
  const int pos = positions[t];
  const int half = head_dim / 2;
  const int q_width = n_heads * head_dim;
  const int kv_width = n_kv_heads * head_dim;
  const int row_width = q_width + 2 * kv_width;
  const __half* src = qkv + static_cast<size_t>(t) * row_width;
  const bool pos_ok = pos >= 0 && pos < max_seq;  // out-of-range rows are dropped, not written

  auto load = [&](int j) {
    float v = __half2float(src[j]);
    // Bias is added in fp16 like the projection output it belongs to.
    if (bias != nullptr) v = __half2float(__hadd(src[j], bias[j]));
    return v;
  };

  for (int p = threadIdx.x; p < (n_heads + n_kv_heads) * half; p += blockDim.x) {
    const int head = p / half;
    const int i = p % half;
    const int j = head * head_dim + i;  // q heads, then k heads, are contiguous in the row
    float x1 = load(j), x2 = load(j + half);
    if (inv_freq != nullptr) {
      float c, s;
      rope_cos_sin(pos, inv_freq[i], c, s);
      const float y1 = x1 * c - x2 * s;
      const float y2 = x2 * c + x1 * s;
      x1 = y1;
      x2 = y2;
    }
    if (head < n_heads) {
      __half* dst = q + static_cast<size_t>(t) * q_width + head * head_dim;
      dst[i] = __float2half(x1);
      dst[i + half] = __float2half(x2);
    } else if (pos_ok) {
      CacheT* dst = k_cache + static_cast<size_t>(pos) * kv_width + (head - n_heads) * head_dim;
      dst[i] = from_float<CacheT>(x1);
      dst[i + half] = from_float<CacheT>(x2);
    }
  }

  if (!pos_ok) return;
  for (int j = threadIdx.x; j < kv_width; j += blockDim.x) {
    v_cache[static_cast<size_t>(pos) * kv_width + j] =
        from_float<CacheT>(load(q_width + kv_width + j));
  }
}

}  // namespace

void split_qkv(const Tensor& qkv, const Tensor& qkv_bias, int n_heads, int n_kv_heads,
               Tensor& q, Tensor& k_cache, Tensor& v_cache, const Tensor& positions,
               const Tensor& inv_freq, cudaStream_t stream) {
  const char* fn = "split_qkv";
  check_cuda(qkv, DType::F16, fn, "qkv");
  check_cuda(q, DType::F16, fn, "q");
  check(k_cache.device() == Device::CUDA &&
            (k_cache.dtype() == DType::F16 || k_cache.dtype() == DType::FP8_E4M3),
        fn, "k_cache must be a CUDA f16 or fp8_e4m3 tensor");
  check(v_cache.dtype() == k_cache.dtype(), fn, "k_cache and v_cache dtypes must match");
  check_cuda(positions, DType::I32, fn, "positions");
  check(q.ndim() == 3 && q.dim(1) == n_heads, fn, "q must be [T, n_heads, head_dim]");
  const int T = static_cast<int>(q.dim(0));
  const int head_dim = static_cast<int>(q.dim(2));
  const int q_width = n_heads * head_dim;
  const int kv_width = n_kv_heads * head_dim;
  check(qkv.ndim() == 2 && qkv.dim(0) == T && qkv.dim(1) == q_width + 2 * kv_width, fn,
        "qkv must be [T, (n_heads + 2*n_kv_heads) * head_dim]");
  check(k_cache.ndim() == 3 && k_cache.dim(1) == n_kv_heads && k_cache.dim(2) == head_dim, fn,
        "k_cache must be [max_seq, n_kv_heads, head_dim]");
  check(v_cache.shape() == k_cache.shape(), fn, "k_cache and v_cache must match");
  check(positions.ndim() == 1 && positions.dim(0) == T, fn, "positions must be [T]");
  if (!qkv_bias.empty()) {
    check_cuda(qkv_bias, DType::F16, fn, "qkv_bias");
    check(qkv_bias.numel() == qkv.dim(1), fn, "qkv_bias must match qkv's last dim");
  }
  check(head_dim % 2 == 0, fn, "head_dim must be even");
  if (!inv_freq.empty()) {
    check_cuda(inv_freq, DType::F32, fn, "inv_freq");
    check(inv_freq.numel() == head_dim / 2, fn, "inv_freq must be [head_dim / 2]");
  }
  if (T == 0) return;

  auto launch = [&](auto* k, auto* v) {
    split_qkv_kernel<<<T, 256, 0, stream>>>(
        qkv.data_as<const __half>(),
        qkv_bias.empty() ? nullptr : qkv_bias.data_as<const __half>(),
        positions.data_as<const int32_t>(),
        inv_freq.empty() ? nullptr : inv_freq.data_as<const float>(), q.data_as<__half>(), k, v,
        n_heads, n_kv_heads, head_dim, static_cast<int>(k_cache.dim(0)));
  };
  if (k_cache.dtype() == DType::F16) {
    launch(k_cache.data_as<__half>(), v_cache.data_as<__half>());
  } else {
    launch(k_cache.data_as<__nv_fp8_e4m3>(), v_cache.data_as<__nv_fp8_e4m3>());
  }
  CUDA_CHECK_LAUNCH();
}

}  // namespace ography::kernels
