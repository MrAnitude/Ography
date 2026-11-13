// Token (and GPT-2 position) embedding lookup. Memory-bound: one block per token.

#include "core/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

namespace {

using namespace detail;

// wpe may be null (plain token embedding).
template <typename OutT>
__global__ void embedding_kernel(const __half* __restrict__ wte, const __half* __restrict__ wpe,
                                 const int32_t* __restrict__ token_ids,
                                 const int32_t* __restrict__ positions, OutT* __restrict__ out,
                                 int hidden, int vocab, int max_pos) {
  const int t = blockIdx.x;
  const int id = token_ids[t];
  OutT* dst = out + static_cast<size_t>(t) * hidden;
  // Out-of-range ids can't throw on the device; emit zeros so the bug is visible in tests
  // instead of reading outside the table.
  if (id < 0 || id >= vocab) {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = from_float<OutT>(0.0f);
    return;
  }
  const __half* row = wte + static_cast<size_t>(id) * hidden;
  const __half* prow = nullptr;
  if (wpe != nullptr) {
    const int p = positions[t];
    if (p >= 0 && p < max_pos) prow = wpe + static_cast<size_t>(p) * hidden;
  }
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = __half2float(row[i]);
    if (prow != nullptr) v += __half2float(prow[i]);
    dst[i] = from_float<OutT>(v);
  }
}

void launch(const Tensor& wte, const Tensor* wpe, const Tensor& token_ids,
            const Tensor* positions, Tensor& out, cudaStream_t stream, const char* fn) {
  check_cuda(wte, DType::F16, fn, "table");
  check_cuda(token_ids, DType::I32, fn, "token_ids");
  check(wte.ndim() == 2 && token_ids.ndim() == 1, fn, "table must be 2-d, token_ids 1-d");
  const int T = static_cast<int>(token_ids.dim(0));
  const int H = static_cast<int>(wte.dim(1));
  check(out.ndim() == 2 && out.dim(0) == T && out.dim(1) == H, fn, "out must be [T, hidden]");
  check(out.dtype() == DType::F16 || out.dtype() == DType::F32, fn, "out must be f16 or f32");
  if (wpe != nullptr) {
    check_cuda(*wpe, DType::F16, fn, "wpe");
    check_cuda(*positions, DType::I32, fn, "positions");
    check(wpe->ndim() == 2 && wpe->dim(1) == H, fn, "wpe must be [max_pos, hidden]");
    check(positions->ndim() == 1 && positions->dim(0) == T, fn, "positions must be [T]");
  }
  if (T == 0) return;

  const __half* wpe_ptr = wpe ? wpe->data_as<const __half>() : nullptr;
  const int32_t* pos_ptr = positions ? positions->data_as<const int32_t>() : nullptr;
  const int max_pos = wpe ? static_cast<int>(wpe->dim(0)) : 0;
  const int vocab = static_cast<int>(wte.dim(0));
  const int threads = 256;
  if (out.dtype() == DType::F32) {
    embedding_kernel<float><<<T, threads, 0, stream>>>(
        wte.data_as<const __half>(), wpe_ptr, token_ids.data_as<const int32_t>(), pos_ptr,
        out.data_as<float>(), H, vocab, max_pos);
  } else {
    embedding_kernel<__half><<<T, threads, 0, stream>>>(
        wte.data_as<const __half>(), wpe_ptr, token_ids.data_as<const int32_t>(), pos_ptr,
        out.data_as<__half>(), H, vocab, max_pos);
  }
  CUDA_CHECK_LAUNCH();
}

}  // namespace

void embedding(const Tensor& table, const Tensor& token_ids, Tensor& out, cudaStream_t stream) {
  launch(table, nullptr, token_ids, nullptr, out, stream, "embedding");
}

void embedding_with_positions(const Tensor& wte, const Tensor& wpe, const Tensor& token_ids,
                              const Tensor& positions, Tensor& out, cudaStream_t stream) {
  launch(wte, &wpe, token_ids, &positions, out, stream, "embedding_with_positions");
}

}  // namespace ography::kernels
