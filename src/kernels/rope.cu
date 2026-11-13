// Rotary position embedding (HF "rotate_half" convention, NOT interleaved pairs).

#include "kernels/kernels.h"
#include "core/logging.h"

namespace ography::kernels {

void rope(Tensor& q, Tensor& k, const Tensor& positions, const Tensor& inv_freq,
          cudaStream_t stream) {
  OGRAPHY_TODO("kernels::rope");
}

}  // namespace ography::kernels
