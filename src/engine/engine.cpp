#include "engine/engine.h"

#include "core/cuda_check.h"
#include "core/logging.h"
#include "core/memory.h"
#include "engine/cuda_graph.h"
#include "kv_cache/kv_cache.h"
#include "kv_cache/prefix_reuse.h"
#include "model/registry.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"

namespace ography {

struct Engine::Impl {
  EngineConfig config;
  // TODO: cudaStream_t stream; cublasHandle_t cublas; std::unique_ptr<DeviceArena> arena;
  //       std::unique_ptr<Model> model; std::unique_ptr<KVCache> kv_cache; PrefixTracker prefix;
  //       Tokenizer tokenizer; ChatTemplate chat_template; DecodeGraph decode_graph;
  //       fixed device buffers: token_ids, positions, seq_len, logits, sampled token.
};

Engine::Engine(const EngineConfig& config) : impl_(std::make_unique<Impl>()) {
  impl_->config = config;
  OGRAPHY_TODO("Engine: create stream/cuBLAS, load model, size KV cache from free VRAM");
}

Engine::~Engine() = default;

GenerationResult Engine::generate(const GenerationRequest& request,
                                  const TokenCallback& on_token) {
  OGRAPHY_TODO("Engine::generate");
}

Tokenizer& Engine::tokenizer() { OGRAPHY_TODO("Engine::tokenizer"); }
ChatTemplate& Engine::chat_template() { OGRAPHY_TODO("Engine::chat_template"); }
const std::string& Engine::model_name() const { OGRAPHY_TODO("Engine::model_name"); }

}  // namespace ography
