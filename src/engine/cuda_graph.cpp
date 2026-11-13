#include "engine/cuda_graph.h"

#include "core/cuda_check.h"
#include "core/logging.h"

namespace ography {

DecodeGraph::~DecodeGraph() { reset(); }

void DecodeGraph::capture(cudaStream_t stream, const std::function<void(cudaStream_t)>& step) {
  OGRAPHY_TODO("DecodeGraph::capture (cudaStreamBeginCapture / EndCapture / GraphInstantiate)");
}

void DecodeGraph::replay(cudaStream_t stream) {
  OGRAPHY_TODO("DecodeGraph::replay (cudaGraphLaunch)");
}

void DecodeGraph::reset() {
  if (exec_) cudaGraphExecDestroy(exec_);
  if (graph_) cudaGraphDestroy(graph_);
  exec_ = nullptr;
  graph_ = nullptr;
}

}  // namespace ography
