// Prefill attention: FlashAttention-2 style tiling with online softmax, causal mask, GQA.

#include "kernels/kernels.h"
#include "core/logging.h"

namespace ography::kernels {

void attention_prefill(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                       int start_pos, float scale, Tensor& out, cudaStream_t stream) {
  OGRAPHY_TODO("kernels::attention_prefill");
}

}  // namespace ography::kernels
