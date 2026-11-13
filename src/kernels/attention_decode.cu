// Decode attention (1 query token). Uses the naive kernel for now.
// TODO: split the sequence across blocks (flash-decoding) so long contexts fill the GPU at
//       batch size 1 (roadmap step 10).

#include "kernels/attention.cuh"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

void attention_decode(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                      const Tensor& positions, float scale, Tensor& out, cudaStream_t stream) {
  detail::check(q.ndim() == 3 && q.dim(0) == 1, "attention_decode", "q must be [1, heads, dim]");
  detail::attention_naive(q, k_cache, v_cache, positions, scale, out, stream, "attention_decode");
}

}  // namespace ography::kernels
