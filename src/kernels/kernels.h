#pragma once

// Host-side launchers for every CUDA kernel. Each launcher lives in its own .cu file
// so kernels can be tested (tests/kernels) and benchmarked (bench/) in isolation.
//
// Conventions:
//   - activations are [tokens, features], row-major, fp16 unless noted
//   - the residual stream is fp32 (cheap at batch 1, and keeps error from accumulating
//     across layers); normalization kernels read it and emit fp16 for the next GEMM
//   - all launches are async on the given stream (CUDA-graph capturable: no syncs, no mallocs)
//   - linear weights use HF layout [out_features, in_features], i.e. y = x @ W^T

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "core/tensor.h"

namespace ography::kernels {

// out[t, :] = table[token_ids[t], :]. table fp16; out fp16 or fp32.
void embedding(const Tensor& table, const Tensor& token_ids, Tensor& out, cudaStream_t stream);

// GPT-2 style: out[t, :] = wte[token_ids[t], :] + wpe[positions[t], :]. out fp16 or fp32.
void embedding_with_positions(const Tensor& wte, const Tensor& wpe, const Tensor& token_ids,
                              const Tensor& positions, Tensor& out, cudaStream_t stream);

// LayerNorm over the last dim: out = (x - mean) / sqrt(var + eps) * weight + bias.
// x [T, H] fp32 or fp16; weight/bias [H] fp16; out [T, H] fp16.
void layernorm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& out,
               cudaStream_t stream);

// Fused residual update + LayerNorm, one pass:
//   residual += x (+ x_bias if not empty);  out = layernorm(residual)
// x [T, H] fp16; x_bias [H] fp16 or empty; residual [T, H] fp32; out [T, H] fp16 (may alias x).
void add_layernorm(const Tensor& x, const Tensor& x_bias, Tensor& residual, const Tensor& weight,
                   const Tensor& bias, float eps, Tensor& out, cudaStream_t stream);

// In place: x = gelu_tanh(x + bias). GPT-2's "gelu_new". x [T, I] fp16; bias [I] fp16.
void bias_gelu(Tensor& x, const Tensor& bias, cudaStream_t stream);

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

// Splits a fused projection qkv [T, (n_heads + 2*n_kv_heads) * head_dim] into q and the KV
// cache, adding the optional fused bias. q: [T, n_heads, head_dim] fp16;
// k_cache/v_cache: [max_seq, n_kv_heads, head_dim] fp16; row t is written at positions[t].
// TODO(llama): apply rope to q/k here, before K lands in the cache.
void split_qkv(const Tensor& qkv, const Tensor& qkv_bias, int n_heads, int n_kv_heads,
               Tensor& q, Tensor& k_cache, Tensor& v_cache, const Tensor& positions,
               cudaStream_t stream);

// Causal attention over the KV cache. Query t attends to cached positions [0, positions[t]],
// so this covers a prefill chunk at any offset and single-token decode alike, and reads
// sequence lengths from device memory (CUDA-graph replayable).
// K/V for these tokens must already be in the cache. Supports GQA (n_heads % n_kv_heads == 0).
// q: [T, n_heads, head_dim]; k_cache/v_cache: [max_seq, n_kv_heads, head_dim]; out like q.
// head_dim must be a multiple of 32, at most 256.
void attention_prefill(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                       const Tensor& positions, float scale, Tensor& out, cudaStream_t stream);

// Single-token decode (T == 1). Same contract as attention_prefill; currently the same naive
// kernel, to be replaced by split-K / flash-decoding.
void attention_decode(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                      const Tensor& positions, float scale, Tensor& out, cudaStream_t stream);

// out = x @ W^T. x [T, in] fp16; weight [out, in] fp16; out [T, out] fp16 or fp32.
// cuBLAS for now (fp32 accumulation); a custom GEMM / GEMV can replace this later.
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
