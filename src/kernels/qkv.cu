// Splits the fused QKV projection into q and the KV cache (adding the fused bias on the way),
// so K/V land in the cache without an extra round trip through memory.

#include "core/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

namespace {

using namespace detail;

__global__ void split_qkv_kernel(const __half* __restrict__ qkv, const __half* __restrict__ bias,
                                 const int32_t* __restrict__ positions, __half* __restrict__ q,
                                 __half* __restrict__ k_cache, __half* __restrict__ v_cache,
                                 int q_width, int kv_width, int max_seq) {
  const int t = blockIdx.x;
  const int pos = positions[t];
  const int row_width = q_width + 2 * kv_width;
  const __half* src = qkv + static_cast<size_t>(t) * row_width;
  const bool pos_ok = pos >= 0 && pos < max_seq;  // out-of-range rows are dropped, not written

  for (int j = threadIdx.x; j < row_width; j += blockDim.x) {
    __half v = src[j];
    if (bias != nullptr) v = __hadd(v, bias[j]);
    if (j < q_width) {
      q[static_cast<size_t>(t) * q_width + j] = v;
    } else if (!pos_ok) {
      continue;
    } else if (j < q_width + kv_width) {
      k_cache[static_cast<size_t>(pos) * kv_width + (j - q_width)] = v;
    } else {
      v_cache[static_cast<size_t>(pos) * kv_width + (j - q_width - kv_width)] = v;
    }
  }
}

}  // namespace

void split_qkv(const Tensor& qkv, const Tensor& qkv_bias, int n_heads, int n_kv_heads,
               Tensor& q, Tensor& k_cache, Tensor& v_cache, const Tensor& positions,
               cudaStream_t stream) {
  const char* fn = "split_qkv";
  check_cuda(qkv, DType::F16, fn, "qkv");
  check_cuda(q, DType::F16, fn, "q");
  check_cuda(k_cache, DType::F16, fn, "k_cache");
  check_cuda(v_cache, DType::F16, fn, "v_cache");
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
  if (T == 0) return;

  split_qkv_kernel<<<T, 256, 0, stream>>>(
      qkv.data_as<const __half>(), qkv_bias.empty() ? nullptr : qkv_bias.data_as<const __half>(),
      positions.data_as<const int32_t>(), q.data_as<__half>(), k_cache.data_as<__half>(),
      v_cache.data_as<__half>(), q_width, kv_width, static_cast<int>(k_cache.dim(0)));
  CUDA_CHECK_LAUNCH();
}

}  // namespace ography::kernels
