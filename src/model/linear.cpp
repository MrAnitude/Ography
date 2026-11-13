#include "model/linear.h"

#include "kernels/kernels.h"

namespace ography {

void linear(cublasHandle_t cublas, const Tensor& x, const QuantizedWeight& w, Tensor& out,
            Tensor& scratch, cudaStream_t stream) {
  if (w.method == QuantMethod::None) {
    kernels::gemm(cublas, x, w.weight, out, stream);
  } else {
    kernels::gemm_w4a16(cublas, x, w.weight, w.scales, w.zeros, w.group_size, out, scratch,
                        stream);
  }
}

}  // namespace ography
