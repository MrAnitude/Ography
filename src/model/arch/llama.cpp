#include "model/arch/llama.h"

#include "core/logging.h"
#include "kernels/kernels.h"

namespace ography {

LlamaModel::LlamaModel(ModelConfig config, ModelWeights weights, DeviceArena& scratch_arena,
                       cublasHandle_t cublas)
    : config_(std::move(config)), weights_(std::move(weights)), cublas_(cublas) {
  // TODO: allocate scratch activations, compute inv_freq_.
}

void LlamaModel::forward(const ForwardInput& input, KVCache& kv_cache, Tensor& logits,
                         cudaStream_t stream) {
  OGRAPHY_TODO("LlamaModel::forward");
}

}  // namespace ography
