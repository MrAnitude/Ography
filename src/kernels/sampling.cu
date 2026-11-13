// Logit penalties and token sampling (greedy, temperature, top-k, top-p, min-p) on the GPU,
// so only a single int32 crosses back to the host per step.
//
// sample() runs as one 1024-thread block. Top-k and top-p thresholds are found by binary
// search over the bit patterns of the (positive) probabilities, one pass over the vocab per
// step; the token is then drawn by an inverse-CDF walk. That is ~60 passes when both filters
// are on (~0.1-0.5 ms at 150K vocab). TODO: radix select if this shows up in profiles.

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <curand_kernel.h>

#include <stdexcept>

#include "core/cuda_check.h"
#include "kernels/kernels.h"

namespace ography::kernels {

namespace {

constexpr int kThreads = 1024;
using BlockReduceF = cub::BlockReduce<float, kThreads>;
using BlockReduceI = cub::BlockReduce<int, kThreads>;
using BlockScanF = cub::BlockScan<float, kThreads>;

struct ArgMax {
  float value;
  int index;
};

struct ArgMaxOp {
  __device__ ArgMax operator()(const ArgMax& a, const ArgMax& b) const {
    if (a.value != b.value) return a.value > b.value ? a : b;
    return a.index < b.index ? a : b;  // ties: lowest id, like torch.argmax
  }
};
using BlockReduceArg = cub::BlockReduce<ArgMax, kThreads>;

union TempStorage {
  BlockReduceF::TempStorage f;
  BlockReduceI::TempStorage i;
  BlockScanF::TempStorage scan;
  BlockReduceArg::TempStorage arg;
};

// Block-wide sum of p >= t (mass) or count of p >= t, broadcast to all threads.
__device__ float mass_at_least(const float* p, int n, float t, TempStorage& tmp, float& bcast) {
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += kThreads) s += p[i] >= t ? p[i] : 0.f;
  s = BlockReduceF(tmp.f).Sum(s);
  if (threadIdx.x == 0) bcast = s;
  __syncthreads();
  const float r = bcast;
  __syncthreads();
  return r;
}

__device__ int count_at_least(const float* p, int n, float t, TempStorage& tmp, int& bcast) {
  int c = 0;
  for (int i = threadIdx.x; i < n; i += kThreads) c += p[i] >= t;
  c = BlockReduceI(tmp.i).Sum(c);
  if (threadIdx.x == 0) bcast = c;
  __syncthreads();
  const int r = bcast;
  __syncthreads();
  return r;
}

// Largest positive float t in (0, 1] with pred(t) true, where pred is monotone (true for small
// t). Positive floats order like their bit patterns, so binary search the bits.
template <typename Pred>
__device__ float search_threshold(Pred pred) {
  uint32_t lo = 0, hi = __float_as_uint(1.0f);
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo + 1) / 2;
    if (pred(__uint_as_float(mid))) lo = mid;
    else hi = mid - 1;
  }
  return __uint_as_float(lo);
}

__global__ void __launch_bounds__(kThreads)
    sample_kernel(const float* __restrict__ logits, int n, SamplingKernelParams host_params,
                  SamplingKernelParams* d_params, float* __restrict__ probs,
                  int32_t* __restrict__ out) {
  // Every thread reads the params before any barrier; thread 0 bumps the step at the end.
  const SamplingKernelParams params = d_params != nullptr ? *d_params : host_params;
  __shared__ TempStorage tmp;
  __shared__ float fb;
  __shared__ int ib;

  // Argmax (also gives the max logit for the softmax).
  ArgMax best{-INFINITY, n};
  for (int i = threadIdx.x; i < n; i += kThreads) best = ArgMaxOp()(best, ArgMax{logits[i], i});
  best = BlockReduceArg(tmp.arg).Reduce(best, ArgMaxOp());
  if (threadIdx.x == 0) {
    fb = best.value;
    ib = best.index;
  }
  __syncthreads();
  const float max_logit = fb;
  const int argmax = ib;
  __syncthreads();

  if (params.temperature <= 0.f || params.top_k == 1) {
    if (threadIdx.x == 0) {
      *out = argmax;
      if (d_params != nullptr) d_params->step = params.step + 1;
    }
    return;
  }

  // Unnormalized probabilities: the max token has p = 1.
  const float inv_t = 1.f / params.temperature;
  for (int i = threadIdx.x; i < n; i += kThreads) probs[i] = __expf((logits[i] - max_logit) * inv_t);
  __syncthreads();

  float threshold = 0.f;
  if (params.top_k > 0 && params.top_k < n) {
    // The k-th largest p: the largest t with at least k tokens >= t.
    threshold = search_threshold(
        [&](float t) { return count_at_least(probs, n, t, tmp, ib) >= params.top_k; });
  }
  if (params.top_p < 1.f) {
    // Smallest set of top tokens holding top_p of what top-k kept.
    const float target = params.top_p * mass_at_least(probs, n, threshold, tmp, fb);
    threshold = fmaxf(threshold, search_threshold([&](float t) {
                        return mass_at_least(probs, n, t, tmp, fb) >= target;
                      }));
  }
  threshold = fmaxf(threshold, params.min_p);  // p_max == 1

  // Inverse CDF over kept tokens: each thread owns a contiguous slice; scan slice sums.
  const int per = (n + kThreads - 1) / kThreads;
  const int begin = min(n, static_cast<int>(threadIdx.x) * per);
  const int end = min(n, begin + per);
  float slice = 0.f;
  for (int i = begin; i < end; ++i) slice += probs[i] >= threshold ? probs[i] : 0.f;
  float prefix, total;
  BlockScanF(tmp.scan).ExclusiveSum(slice, prefix, total);

  if (threadIdx.x == 0) {
    curandStatePhilox4_32_10_t rng;
    curand_init(params.seed, params.step, 0, &rng);
    fb = (1.f - curand_uniform(&rng)) * total;  // curand_uniform is (0, 1]
    ib = argmax;  // fallback if rounding leaves u past every slice
  }
  __syncthreads();
  const float u = fb;
  if (slice > 0.f && u >= prefix && u < prefix + slice) {
    float acc = prefix;
    int pick = -1;
    for (int i = begin; i < end; ++i) {
      if (probs[i] < threshold) continue;
      pick = i;  // last kept token in the slice, if rounding runs past the end
      acc += probs[i];
      if (u < acc) break;
    }
    ib = pick;  // exactly one thread's slice contains u
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    *out = ib;
    if (d_params != nullptr) d_params->step = params.step + 1;
  }
}

__global__ void penalties_kernel(float* __restrict__ logits, const int8_t* __restrict__ prompt,
                                 const int32_t* __restrict__ counts, int n, float repetition,
                                 float presence, float frequency,
                                 const SamplingKernelParams* __restrict__ d_params) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (d_params != nullptr) {
    repetition = d_params->repetition;
    presence = d_params->presence;
    frequency = d_params->frequency;
  }
  if (i >= n || (repetition == 1.f && presence == 0.f && frequency == 0.f)) return;
  const int c = counts[i];
  float l = logits[i];
  if (repetition != 1.f && (c > 0 || prompt[i])) l = l > 0.f ? l / repetition : l * repetition;
  if (c > 0) l -= presence + frequency * static_cast<float>(c);
  logits[i] = l;
}

__global__ void mark_kernel(const int32_t* __restrict__ ids, int n, int8_t* __restrict__ mask) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) mask[ids[i]] = 1;
}

__global__ void count_kernel(const int32_t* __restrict__ token, int32_t* __restrict__ counts) {
  counts[*token] += 1;
}

void check_vocab(const Tensor& logits, const Tensor& other, DType dtype, const char* what) {
  if (other.dtype() != dtype || other.numel() < logits.numel()) {
    throw std::invalid_argument(std::string(what) + ": wrong dtype or smaller than vocab");
  }
}

}  // namespace

void sample(const Tensor& logits, const SamplingKernelParams& params, int32_t* d_out_token,
            Tensor& scratch, cudaStream_t stream) {
  if (logits.dtype() != DType::F32) throw std::invalid_argument("sample: logits must be f32");
  check_vocab(logits, scratch, DType::F32, "sample: scratch");
  sample_kernel<<<1, kThreads, 0, stream>>>(logits.data_as<const float>(),
                                            static_cast<int>(logits.numel()), params, nullptr,
                                            scratch.data_as<float>(), d_out_token);
  CUDA_CHECK_LAUNCH();
}

void sample(const Tensor& logits, SamplingKernelParams* d_params, int32_t* d_out_token,
            Tensor& scratch, cudaStream_t stream) {
  if (logits.dtype() != DType::F32) throw std::invalid_argument("sample: logits must be f32");
  check_vocab(logits, scratch, DType::F32, "sample: scratch");
  sample_kernel<<<1, kThreads, 0, stream>>>(logits.data_as<const float>(),
                                            static_cast<int>(logits.numel()),
                                            SamplingKernelParams{}, d_params,
                                            scratch.data_as<float>(), d_out_token);
  CUDA_CHECK_LAUNCH();
}

void apply_penalties(Tensor& logits, const Tensor& prompt_mask, const Tensor& output_counts,
                     float repetition, float presence, float frequency, cudaStream_t stream) {
  check_vocab(logits, prompt_mask, DType::I8, "apply_penalties: prompt_mask");
  check_vocab(logits, output_counts, DType::I32, "apply_penalties: output_counts");
  const int n = static_cast<int>(logits.numel());
  penalties_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      logits.data_as<float>(), prompt_mask.data_as<const int8_t>(),
      output_counts.data_as<const int32_t>(), n, repetition, presence, frequency, nullptr);
  CUDA_CHECK_LAUNCH();
}

void apply_penalties(Tensor& logits, const Tensor& prompt_mask, const Tensor& output_counts,
                     const SamplingKernelParams* d_params, cudaStream_t stream) {
  check_vocab(logits, prompt_mask, DType::I8, "apply_penalties: prompt_mask");
  check_vocab(logits, output_counts, DType::I32, "apply_penalties: output_counts");
  const int n = static_cast<int>(logits.numel());
  penalties_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      logits.data_as<float>(), prompt_mask.data_as<const int8_t>(),
      output_counts.data_as<const int32_t>(), n, 1.f, 0.f, 0.f, d_params);
  CUDA_CHECK_LAUNCH();
}

void mark_tokens(const Tensor& ids, Tensor& prompt_mask, cudaStream_t stream) {
  const int n = static_cast<int>(ids.numel());
  if (n == 0) return;
  mark_kernel<<<(n + 255) / 256, 256, 0, stream>>>(ids.data_as<const int32_t>(), n,
                                                   prompt_mask.data_as<int8_t>());
  CUDA_CHECK_LAUNCH();
}

void count_token(const int32_t* d_token, Tensor& output_counts, cudaStream_t stream) {
  count_kernel<<<1, 1, 0, stream>>>(d_token, output_counts.data_as<int32_t>());
  CUDA_CHECK_LAUNCH();
}

}  // namespace ography::kernels
