#pragma once

#include <cuda_runtime.h>

#include "core/tensor.h"
#include "kv_cache/kv_cache.h"
#include "model/config.h"

namespace ography {

// Inputs for one forward pass over n_tokens new tokens (prefill chunk, or 1 for decode).
// All tensors are device tensors in pre-allocated, fixed buffers so decode can be
// captured into a CUDA graph.
struct ForwardInput {
  Tensor token_ids;     // [n_tokens] int32
  Tensor positions;     // [n_tokens] int32, absolute positions
  int n_tokens = 0;
  int start_pos = 0;    // == kv_cache.seq_len() before this pass
  bool is_decode = false;
};

// A decoder-only transformer. Implementations live in model/arch/.
class Model {
 public:
  virtual ~Model() = default;

  // Runs the forward pass, appends K/V to the cache, and writes logits for the
  // LAST token only ([vocab]); earlier positions' logits are never needed.
  virtual void forward(const ForwardInput& input, KVCache& kv_cache, Tensor& logits,
                       cudaStream_t stream) = 0;

  virtual const ModelConfig& config() const = 0;
};

}  // namespace ography
