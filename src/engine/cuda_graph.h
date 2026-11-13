#pragma once

#include <functional>

#include <cuda_runtime.h>

namespace ography {

// Captures one decode step (the whole forward pass + sampling for 1 token) into a CUDA
// graph and replays it every step. At batch size 1, launch overhead from ~hundreds of
// small kernels dominates, so this is one of the largest latency wins.
//
// Requirements on the captured code: fixed buffer addresses, no host syncs, no allocations,
// and anything that changes per step (position, seq_len) read from device memory.
class DecodeGraph {
 public:
  ~DecodeGraph();

  void capture(cudaStream_t stream, const std::function<void(cudaStream_t)>& step);
  void replay(cudaStream_t stream);
  bool captured() const { return exec_ != nullptr; }
  void reset();

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
};

}  // namespace ography
