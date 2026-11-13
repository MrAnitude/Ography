#include "engine/cuda_graph.h"

#include "core/cuda_check.h"

namespace ography {

DecodeGraph::~DecodeGraph() { reset(); }

void DecodeGraph::capture(cudaStream_t stream, const std::function<void(cudaStream_t)>& step) {
  reset();
  // Thread-local mode: only this thread's CUDA calls are checked for capture safety, so
  // other threads (e.g. HTTP handlers) aren't affected.
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  try {
    step(stream);
  } catch (...) {
    cudaGraph_t partial = nullptr;
    cudaStreamEndCapture(stream, &partial);
    if (partial) cudaGraphDestroy(partial);
    throw;
  }
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
  CUDA_CHECK(cudaGraphInstantiate(&exec_, graph_, 0));
}

void DecodeGraph::replay(cudaStream_t stream) { CUDA_CHECK(cudaGraphLaunch(exec_, stream)); }

void DecodeGraph::reset() {
  if (exec_) cudaGraphExecDestroy(exec_);
  if (graph_) cudaGraphDestroy(graph_);
  exec_ = nullptr;
  graph_ = nullptr;
}

}  // namespace ography
