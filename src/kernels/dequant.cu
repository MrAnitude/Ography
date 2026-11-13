// INT4 weights: checkpoint repacking, dequantization, and the W4A16 GEMV used for decode.

#include <algorithm>

#include "core/cuda_check.h"
#include "kernels/common.cuh"
#include "kernels/kernels.h"

namespace ography::kernels {

namespace {

using namespace detail;

// AWQ packs output column 8b+j into nibble kAwqNibble[j] of word b.
__constant__ int kAwqNibble[8] = {0, 4, 1, 5, 2, 6, 3, 7};

__device__ __forceinline__ uint32_t nibble(uint32_t word, int k) { return (word >> (4 * k)) & 0xF; }

// One thread per output word (o, b) of the engine layout.
__global__ void repack_kernel(Int4Source source, const uint32_t* __restrict__ qweight, int in,
                              int out, uint32_t* __restrict__ dst, int row_offset) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int words_per_row = in / 8;
  if (idx >= static_cast<size_t>(out) * words_per_row) return;
  const int o = static_cast<int>(idx / words_per_row);
  const int b = static_cast<int>(idx % words_per_row);
  uint32_t packed = 0;
  for (int k = 0; k < 8; ++k) {
    const int i = 8 * b + k;
    uint32_t q;
    if (source == Int4Source::AWQ) {
      q = nibble(qweight[static_cast<size_t>(i) * (out / 8) + o / 8], kAwqNibble[o % 8]);
    } else {
      q = nibble(qweight[static_cast<size_t>(i / 8) * out + o], i % 8);
    }
    packed |= q << (4 * k);
  }
  dst[static_cast<size_t>(row_offset + o) * words_per_row + b] = packed;
}

// One thread per (group, column): transposes scales and unpacks zeros.
__global__ void repack_scales_kernel(Int4Source source, const uint32_t* __restrict__ qzeros,
                                     const __half* __restrict__ scales, int groups, int out,
                                     __half* __restrict__ dst_scales,
                                     __half* __restrict__ dst_zeros, int row_offset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= groups * out) return;
  const int g = idx / out;
  const int o = idx % out;
  const uint32_t word = qzeros[static_cast<size_t>(g) * (out / 8) + o / 8];
  const int z = source == Int4Source::AWQ ? nibble(word, kAwqNibble[o % 8])
                                          : nibble(word, o % 8) + 1;  // GPTQ stores zero - 1
  const size_t dst = static_cast<size_t>(row_offset + o) * groups + g;
  dst_scales[dst] = scales[static_cast<size_t>(g) * out + o];
  dst_zeros[dst] = __int2half_rn(z);
}

// One thread per packed word: 8 fp16 outputs.
__global__ void dequant_kernel(const uint32_t* __restrict__ qweight,
                               const __half* __restrict__ scales,
                               const __half* __restrict__ zeros, int in, int groups,
                               int group_size, int row_begin, int rows, __half* __restrict__ out) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int words_per_row = in / 8;
  if (idx >= static_cast<size_t>(rows) * words_per_row) return;
  const int r = static_cast<int>(idx / words_per_row);
  const int b = static_cast<int>(idx % words_per_row);
  const size_t row = static_cast<size_t>(row_begin + r);
  const uint32_t word = qweight[row * words_per_row + b];
  const int g = 8 * b / group_size;
  const float s = __half2float(scales[row * groups + g]);
  const float z = __half2float(zeros[row * groups + g]);
  __half vals[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) vals[k] = __float2half((static_cast<float>(nibble(word, k)) - z) * s);
  *reinterpret_cast<uint4*>(out + static_cast<size_t>(r) * in + 8 * b) =
      *reinterpret_cast<const uint4*>(vals);
}

constexpr int kGemvWarps = 8;  // output rows per block

// y[o] = sum_i W[o, i] x[i], one warp per output row. x lives in shared memory (fp16). Each
// lane loads 16 bytes of packed weights (32 inputs) per step, so a warp streams 512
// contiguous bytes of its row per iteration. Per group of inputs:
//   sum_i (q_i - z) s x_i = s * (sum_i q_i x_i - z * sum_i x_i)
__global__ void __launch_bounds__(kGemvWarps * 32)
    gemv_w4a16_kernel(const __half* __restrict__ x, const uint4* __restrict__ qweight,
                      const __half* __restrict__ scales, const __half* __restrict__ zeros,
                      int in, int out, int groups, int group_size, __half* __restrict__ y) {
  extern __shared__ __half s_x[];
  for (int i = threadIdx.x * 8; i < in; i += blockDim.x * 8) {
    *reinterpret_cast<uint4*>(s_x + i) = *reinterpret_cast<const uint4*>(x + i);
  }
  __syncthreads();

  const int lane = threadIdx.x % 32;
  const int o = blockIdx.x * kGemvWarps + threadIdx.x / 32;
  if (o >= out) return;
  const int chunks = in / 32;  // 16-byte chunks per row
  const uint4* row = qweight + static_cast<size_t>(o) * chunks;
  const __half* row_s = scales + static_cast<size_t>(o) * groups;
  const __half* row_z = zeros + static_cast<size_t>(o) * groups;

  float acc = 0.f;
  for (int c = lane; c < chunks; c += 32) {
    const uint4 w = row[c];
    const uint32_t words[4] = {w.x, w.y, w.z, w.w};
    const __half2* xs = reinterpret_cast<const __half2*>(s_x + 32 * c);
    float qx = 0.f, sx = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
      for (int k = 0; k < 8; k += 2) {
        const float2 xv = __half22float2(xs[j * 4 + k / 2]);
        qx += static_cast<float>(nibble(words[j], k)) * xv.x +
              static_cast<float>(nibble(words[j], k + 1)) * xv.y;
        sx += xv.x + xv.y;
      }
    }
    const int g = 32 * c / group_size;
    acc += __half2float(row_s[g]) * (qx - __half2float(row_z[g]) * sx);
  }
  acc = warp_sum(acc);
  if (lane == 0) y[o] = __float2half(acc);
}

void check_int4(const Tensor& qweight, const Tensor& scales, const Tensor& zeros, int group_size,
                const char* fn) {
  check_cuda(qweight, DType::I32, fn, "qweight");
  check_cuda(scales, DType::F16, fn, "scales");
  check_cuda(zeros, DType::F16, fn, "zeros");
  check(qweight.ndim() == 2 && scales.ndim() == 2 && zeros.shape() == scales.shape(), fn,
        "qweight [out, in/8], scales/zeros [out, groups]");
  const int64_t in = qweight.dim(1) * 8;
  check(group_size > 0 && group_size % 32 == 0 && in % group_size == 0, fn,
        "group_size must be a multiple of 32 dividing in");
  check(scales.dim(0) == qweight.dim(0) && scales.dim(1) == in / group_size, fn,
        "scales must be [out, in / group_size]");
}

}  // namespace

void repack_int4(Int4Source source, const Tensor& qweight, const Tensor& qzeros,
                 const Tensor& scales, int group_size, Tensor& out_qweight, Tensor& out_scales,
                 Tensor& out_zeros, int row_offset, cudaStream_t stream) {
  const char* fn = "repack_int4";
  check_cuda(qweight, DType::I32, fn, "qweight");
  check_cuda(qzeros, DType::I32, fn, "qzeros");
  check_cuda(scales, DType::F16, fn, "scales");
  check(scales.ndim() == 2, fn, "scales must be [groups, out]");
  const int groups = static_cast<int>(scales.dim(0));
  const int out = static_cast<int>(scales.dim(1));
  const int in = groups * group_size;
  check(out % 8 == 0 && qzeros.ndim() == 2 && qzeros.dim(0) == groups && qzeros.dim(1) == out / 8,
        fn, "qzeros must be [groups, out/8]");
  if (source == Int4Source::AWQ) {
    check(qweight.ndim() == 2 && qweight.dim(0) == in && qweight.dim(1) == out / 8, fn,
          "AWQ qweight must be [in, out/8]");
  } else {
    check(qweight.ndim() == 2 && qweight.dim(0) == in / 8 && qweight.dim(1) == out, fn,
          "GPTQ qweight must be [in/8, out]");
  }
  check_int4(out_qweight, out_scales, out_zeros, group_size, fn);
  check(out_qweight.dim(1) == in / 8 && row_offset >= 0 && row_offset + out <= out_qweight.dim(0),
        fn, "destination rows out of range or input size mismatch");

  const size_t words = static_cast<size_t>(out) * (in / 8);
  repack_kernel<<<static_cast<unsigned>((words + 255) / 256), 256, 0, stream>>>(
      source, qweight.data_as<const uint32_t>(), in, out, out_qweight.data_as<uint32_t>(),
      row_offset);
  CUDA_CHECK_LAUNCH();
  repack_scales_kernel<<<(groups * out + 255) / 256, 256, 0, stream>>>(
      source, qzeros.data_as<const uint32_t>(), scales.data_as<const __half>(), groups, out,
      out_scales.data_as<__half>(), out_zeros.data_as<__half>(), row_offset);
  CUDA_CHECK_LAUNCH();
}

void dequant_int4(const Tensor& qweight, const Tensor& scales, const Tensor& zeros,
                  int group_size, int row_begin, Tensor& out, cudaStream_t stream) {
  const char* fn = "dequant_int4";
  check_int4(qweight, scales, zeros, group_size, fn);
  check_cuda(out, DType::F16, fn, "out");
  const int in = static_cast<int>(qweight.dim(1) * 8);
  const int rows = static_cast<int>(out.dim(0));
  check(out.ndim() == 2 && out.dim(1) == in && row_begin >= 0 && row_begin + rows <= qweight.dim(0),
        fn, "out must be [rows, in] within the weight");
  const size_t words = static_cast<size_t>(rows) * (in / 8);
  if (words == 0) return;
  dequant_kernel<<<static_cast<unsigned>((words + 255) / 256), 256, 0, stream>>>(
      qweight.data_as<const uint32_t>(), scales.data_as<const __half>(),
      zeros.data_as<const __half>(), in, static_cast<int>(scales.dim(1)), group_size, row_begin,
      rows, out.data_as<__half>());
  CUDA_CHECK_LAUNCH();
}

void gemm_w4a16(cublasHandle_t handle, const Tensor& x, const Tensor& qweight,
                const Tensor& scales, const Tensor& zeros, int group_size, Tensor& out,
                Tensor& scratch, cudaStream_t stream) {
  const char* fn = "gemm_w4a16";
  check_int4(qweight, scales, zeros, group_size, fn);
  check_cuda(x, DType::F16, fn, "x");
  check_cuda(out, DType::F16, fn, "out");
  const int T = static_cast<int>(x.dim(0));
  const int in = static_cast<int>(x.dim(1));
  const int n_out = static_cast<int>(qweight.dim(0));
  check(x.ndim() == 2 && qweight.dim(1) * 8 == in, fn, "x must be [T, in]");
  check(out.ndim() == 2 && out.dim(0) == T && out.dim(1) == n_out, fn, "out must be [T, out]");
  if (T == 0) return;

  if (T == 1) {
    const size_t smem = static_cast<size_t>(in) * sizeof(__half);
    check(smem <= 96 * 1024, fn, "in too large for the GEMV's shared-memory x");
    if (smem > 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(gemv_w4a16_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)));
    }
    gemv_w4a16_kernel<<<(n_out + kGemvWarps - 1) / kGemvWarps, kGemvWarps * 32, smem, stream>>>(
        x.data_as<const __half>(), qweight.data_as<const uint4>(), scales.data_as<const __half>(),
        zeros.data_as<const __half>(), in, n_out, static_cast<int>(scales.dim(1)), group_size,
        out.data_as<__half>());
    CUDA_CHECK_LAUNCH();
    return;
  }

  // Prefill: dequantize as many rows as fit in scratch, multiply, repeat. Each tile of W rows
  // produces the matching columns of out (row-major [T, out] == column-major [out, T], so a
  // column block is a pointer offset with ldc = n_out).
  check_cuda(scratch, DType::F16, fn, "scratch");
  const int tile_rows = static_cast<int>(std::min<int64_t>(n_out, scratch.numel() / in));
  check(tile_rows > 0, fn, "scratch must hold at least one row");
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  for (int r0 = 0; r0 < n_out; r0 += tile_rows) {
    const int rows = std::min(tile_rows, n_out - r0);
    Tensor tile(scratch.data(), DType::F16, {rows, in}, Device::CUDA);
    dequant_int4(qweight, scales, zeros, group_size, r0, tile, stream);
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, T, in, &alpha, tile.data(),
                              CUDA_R_16F, in, x.data(), CUDA_R_16F, in, &beta,
                              out.data_as<__half>() + r0, CUDA_R_16F, n_out, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
  }
}

}  // namespace ography::kernels
