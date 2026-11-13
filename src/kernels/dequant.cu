// Weight dequantization kernels for quantized checkpoints.

#include "kernels/kernels.h"
#include "core/logging.h"

namespace ography::kernels {

void dequant_int4(const Tensor& qweight, const Tensor& scales, const Tensor& zeros,
                  int group_size, Tensor& out, cudaStream_t stream) {
  OGRAPHY_TODO("kernels::dequant_int4");
}

}  // namespace ography::kernels
