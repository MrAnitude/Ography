#pragma once

#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// Abort on any CUDA / cuBLAS error. The engine treats GPU errors as fatal.
#define CUDA_CHECK(expr)                                                          \
  do {                                                                            \
    cudaError_t err_ = (expr);                                                    \
    if (err_ != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), \
                   __FILE__, __LINE__, cudaGetErrorString(err_));                 \
      std::abort();                                                               \
    }                                                                             \
  } while (0)

#define CUBLAS_CHECK(expr)                                                      \
  do {                                                                          \
    cublasStatus_t st_ = (expr);                                                \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                         \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(st_), \
                   __FILE__, __LINE__);                                         \
      std::abort();                                                             \
    }                                                                           \
  } while (0)

// Catch asynchronous kernel launch errors right after a launch.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
