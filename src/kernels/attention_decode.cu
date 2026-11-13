// Decode attention (1 query token) and KV cache writes. Split the sequence across blocks
// (flash-decoding) so long contexts still fill the GPU at batch size 1.

#include "kernels/kernels.h"
#include "core/logging.h"

namespace ography::kernels {

void attention_decode(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                      const int32_t* d_seq_len, float scale, Tensor& out, cudaStream_t stream) {
  OGRAPHY_TODO("kernels::attention_decode");
}

void kv_cache_write(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache,
                    const Tensor& positions, cudaStream_t stream) {
  OGRAPHY_TODO("kernels::kv_cache_write");
}

}  // namespace ography::kernels
