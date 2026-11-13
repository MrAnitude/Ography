#pragma once

// Host-side launchers for every CUDA kernel. Each launcher lives in its own .cu file
// so kernels can be tested (tests/kernels) and benchmarked (bench/) in isolation.
//
// Conventions:
//   - activations are [tokens, features], row-major, fp16/bf16 unless noted
//   - all launches are async on the given stream (CUDA-graph capturable: no syncs, no mallocs)
//   - linear weights use HF layout [out_features, in_features], i.e. y = x @ W^T

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "core/tensor.h"

namespace ography::kernels {

// out[t, :] = table[token_ids[t], :]
void embedding(const Tensor& table, const Tensor& token_ids, Tensor& out, cudaStream_t stream);

// out = x / rms(x) * weight
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out, cudaStream_t stream);

// Fused: residual += x; x = rmsnorm(residual) * weight. One pass over memory instead of two.
void add_rmsnorm(Tensor& x, Tensor& residual, const Tensor& weight, float eps,
                 cudaStream_t stream);

// In-place rotary embedding on q [T, n_heads, head_dim] and k [T, n_kv_heads, head_dim].
// TODO: rope_scaling variants (llama3, yarn) via precomputed inv_freq table.
void rope(Tensor& q, Tensor& k, const Tensor& positions, const Tensor& inv_freq,
          cudaStream_t stream);

// gate_up [T, 2*I] (gate and up projections fused) -> out [T, I] = silu(gate) * up
void swiglu(const Tensor& gate_up, Tensor& out, cudaStream_t stream);

// Causal attention for a prompt chunk of T tokens starting at absolute position start_pos.
// New K/V must already be written to the cache. Supports GQA (n_heads % n_kv_heads == 0).
// q: [T, n_heads, head_dim]; k_cache/v_cache: [max_seq, n_kv_heads, head_dim]; out like q.
void attention_prefill(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                       int start_pos, float scale, Tensor& out, cudaStream_t stream);

// Single-token decode attention over seq_len cached positions (split-K / flash-decoding).
// seq_len is read from device memory so the launch is CUDA-graph replayable.
void attention_decode(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                      const int32_t* d_seq_len, float scale, Tensor& out, cudaStream_t stream);

// Writes this step's K/V rows into the cache at the given positions.
void kv_cache_write(const Tensor& k, const Tensor& v, Tensor& k_cache, Tensor& v_cache,
                    const Tensor& positions, cudaStream_t stream);

// out = x @ W^T. cuBLAS for now; a custom GEMM / GEMV can replace this later.
void gemm(cublasHandle_t handle, const Tensor& x, const Tensor& weight, Tensor& out,
          cudaStream_t stream);

struct SamplingKernelParams {
  float temperature = 1.0f;  // 0 => greedy
  int top_k = 0;             // 0 => disabled
  float top_p = 1.0f;
  float min_p = 0.0f;
  uint64_t seed = 0;
  uint64_t step = 0;  // combined with seed for the RNG stream
};

// logits [vocab] -> one token id written to d_out_token. Penalties are applied beforehand.
void sample(const Tensor& logits, const SamplingKernelParams& params, int32_t* d_out_token,
            Tensor& scratch, cudaStream_t stream);

// Repetition / presence / frequency penalties applied in-place to logits.
void apply_penalties(Tensor& logits, const Tensor& token_counts, float repetition,
                     float presence, float frequency, cudaStream_t stream);

// Group-wise INT4 -> fp16 dequantization (AWQ/GPTQ-style). Baseline before a fused W4A16 GEMM.
void dequant_int4(const Tensor& qweight, const Tensor& scales, const Tensor& zeros,
                  int group_size, Tensor& out, cudaStream_t stream);

}  // namespace ography::kernels
