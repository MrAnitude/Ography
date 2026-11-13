// Token embedding lookup. Memory-bound: one block per token, vectorized row copy.

#include "kernels/kernels.h"
#include "core/logging.h"

namespace ography::kernels {

void embedding(const Tensor& table, const Tensor& token_ids, Tensor& out, cudaStream_t stream) {
  OGRAPHY_TODO("kernels::embedding");
}

}  // namespace ography::kernels
