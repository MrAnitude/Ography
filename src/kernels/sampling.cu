// Logit penalties and token sampling (greedy, temperature, top-k, top-p, min-p) on the GPU,
// so only a single int32 crosses back to the host per step.

#include "kernels/kernels.h"
#include "core/logging.h"

namespace ography::kernels {

void sample(const Tensor& logits, const SamplingKernelParams& params, int32_t* d_out_token,
            Tensor& scratch, cudaStream_t stream) {
  OGRAPHY_TODO("kernels::sample");
}

void apply_penalties(Tensor& logits, const Tensor& token_counts, float repetition,
                     float presence, float frequency, cudaStream_t stream) {
  OGRAPHY_TODO("kernels::apply_penalties");
}

}  // namespace ography::kernels
