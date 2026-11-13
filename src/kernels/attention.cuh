#pragma once

// Internal attention launchers shared between attention_prefill.cu and attention_decode.cu.

#include <cuda_runtime.h>

#include "core/tensor.h"

namespace ography::kernels::detail {

// Validates arguments against the contract documented on kernels::attention_prefill.
void check_attention(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                     const Tensor& positions, const Tensor& out, const char* fn);

// Naive causal attention (any head_dim that is a multiple of 32, <= 256).
void attention_naive(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                     const Tensor& positions, float scale, Tensor& out, cudaStream_t stream,
                     const char* fn);

// Tensor-core flash attention (attention_flash.cu). Returns false, launching nothing, for
// head sizes it doesn't handle (it does 64 and 128).
bool attention_flash(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                     const Tensor& positions, float scale, Tensor& out, cudaStream_t stream);

}  // namespace ography::kernels::detail
