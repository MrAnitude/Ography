// RMSNorm and fused residual-add + RMSNorm. One block per row, warp-shuffle reduction, fp32 accumulation.

#include "kernels/kernels.h"
#include "core/logging.h"

namespace ography::kernels {

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out, cudaStream_t stream) {
  OGRAPHY_TODO("kernels::rmsnorm");
}

void add_rmsnorm(Tensor& x, Tensor& residual, const Tensor& weight, float eps,
                 cudaStream_t stream) {
  OGRAPHY_TODO("kernels::add_rmsnorm");
}

}  // namespace ography::kernels
