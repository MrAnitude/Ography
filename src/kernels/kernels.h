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

// RMSNorm over the last dim: out = x / sqrt(mean(x^2) + eps) * weight (fp32 statistics).
// x [T, H] fp32 or fp16; weight [H] fp16; out [T, H] fp16.
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out, cudaStream_t stream);

// Fused residual update + RMSNorm, one pass:  residual += x;  out = rmsnorm(residual)
// x [T, H] fp16; residual [T, H] fp32; out [T, H] fp16 (may alias x).
void add_rmsnorm(const Tensor& x, Tensor& residual, const Tensor& weight, float eps, Tensor& out,
                 cudaStream_t stream);

// gate_up [T, 2*I] (gate and up projections fused) -> out [T, I] = silu(gate) * up. fp16.
void swiglu(const Tensor& gate_up, Tensor& out, cudaStream_t stream);

// Splits a fused projection qkv [T, (n_heads + 2*n_kv_heads) * head_dim] into q and the KV
// cache, adding the optional fused bias and applying rotary embeddings to q and k when
// inv_freq is given, so K lands in the cache already rotated.
// q: [T, n_heads, head_dim] fp16; k_cache/v_cache: [max_seq, n_kv_heads, head_dim] fp16;
// row t is written at positions[t]. inv_freq: [head_dim / 2] fp32 (rope scaling already
// applied), or empty for no rope. RoPE uses HF's rotate_half convention (pairs i, i + d/2).
void split_qkv(const Tensor& qkv, const Tensor& qkv_bias, int n_heads, int n_kv_heads,
               Tensor& q, Tensor& k_cache, Tensor& v_cache, const Tensor& positions,
               const Tensor& inv_freq, cudaStream_t stream);

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
  float temperature = 1.0f;  // <= 0 => greedy
  int top_k = 0;             // 0 => disabled
  float top_p = 1.0f;        // 1 => disabled
  float min_p = 0.0f;        // 0 => disabled; keeps p >= min_p * p_max
  uint64_t seed = 0;
  uint64_t step = 0;  // Philox subsequence: one independent random draw per (seed, step)
};

// logits [vocab] f32 -> one token id written to d_out_token. Filters apply in HF order:
// temperature, top-k, top-p (over what top-k kept), min-p. Greedy (or top_k == 1) is an
// argmax with ties going to the lowest id. scratch: f32, at least [vocab]. Penalties are
// applied beforehand with apply_penalties. One block; no host sync.
void sample(const Tensor& logits, const SamplingKernelParams& params, int32_t* d_out_token,
            Tensor& scratch, cudaStream_t stream);

// In-place on logits [vocab] f32, as in vLLM:
//   repetition (HF-style, divides positive / multiplies negative logits) for tokens seen in
//   the prompt (prompt_mask [vocab] i8 != 0) or the output (output_counts [vocab] i32 > 0);
//   presence and frequency (OpenAI-style, subtracted) from output_counts only.
void apply_penalties(Tensor& logits, const Tensor& prompt_mask, const Tensor& output_counts,
                     float repetition, float presence, float frequency, cudaStream_t stream);

// prompt_mask[ids[i]] = 1 for each prompt token (mask must be zeroed first).
void mark_tokens(const Tensor& ids, Tensor& prompt_mask, cudaStream_t stream);
// output_counts[*d_token] += 1, reading the token from device memory (graph-safe).
void count_token(const int32_t* d_token, Tensor& output_counts, cudaStream_t stream);

// Group-wise INT4 -> fp16 dequantization (AWQ/GPTQ-style). Baseline before a fused W4A16 GEMM.
void dequant_int4(const Tensor& qweight, const Tensor& scales, const Tensor& zeros,
                  int group_size, Tensor& out, cudaStream_t stream);

}  // namespace ography::kernels
