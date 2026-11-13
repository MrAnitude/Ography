#pragma once

// Internal attention launchers shared between attention_prefill.cu and attention_decode.cu.

#include <cuda_runtime.h>

#include "core/tensor.h"

namespace ography::kernels::detail {

// Naive causal attention, contract as documented on kernels::attention_prefill.
void attention_naive(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                     const Tensor& positions, float scale, Tensor& out, cudaStream_t stream,
                     const char* fn);

}  // namespace ography::kernels::detail
