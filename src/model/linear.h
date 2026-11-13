#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "core/tensor.h"
#include "quant/quant_config.h"

namespace ography {

// out = x @ W^T for a linear weight in any supported storage: fp16 (cuBLAS) or INT4 (fused
// GEMV for decode, tiled dequant + cuBLAS for prefill, using scratch).
void linear(cublasHandle_t cublas, const Tensor& x, const QuantizedWeight& w, Tensor& out,
            Tensor& scratch, cudaStream_t stream);

}  // namespace ography
