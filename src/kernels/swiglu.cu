// SwiGLU activation: silu(gate) * up over the fused gate_up projection output.

#include "kernels/kernels.h"
#include "core/logging.h"

namespace ography::kernels {

void swiglu(const Tensor& gate_up, Tensor& out, cudaStream_t stream) {
  OGRAPHY_TODO("kernels::swiglu");
}

}  // namespace ography::kernels
