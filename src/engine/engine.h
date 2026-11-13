#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "engine/sampling_params.h"
#include "engine/stop_checker.h"

namespace ography {

class Tokenizer;
class ChatTemplate;

struct EngineConfig {
  std::string model_dir;            // HF model directory (config.json, *.safetensors, tokenizer.json)
  int max_seq_len = 0;              // 0 => as large as free VRAM allows (capped at model max)
  bool fp8_kv_cache = false;
  bool use_cuda_graphs = true;
  int prefill_chunk = 2048;         // longest prompt slice per forward pass (bounds scratch memory)
  float vram_fraction = 0.92f;      // how much free VRAM the engine may claim
};

struct GenerationRequest {
  std::vector<int32_t> prompt_tokens;
  SamplingParams sampling;
};

struct TokenEvent {
  int32_t token_id;
  std::string text;  // UTF-8-complete text to stream (may be empty)
};

struct GenerationResult {
  FinishReason finish_reason = FinishReason::Stop;
  int prompt_tokens = 0;
  int completion_tokens = 0;     // includes a final stop token, as OpenAI counts it
  int cached_prompt_tokens = 0;  // reused from the previous turn's KV cache
  double time_to_first_token_ms = 0;
  double tokens_per_second = 0;  // decode rate, after the first token
};

// Return false to cancel (e.g. the HTTP client disconnected).
using TokenCallback = std::function<bool(const TokenEvent&)>;

// Owns the GPU: model, KV cache, memory arena, CUDA stream, cuBLAS handle, decode graph.
// Runs one request at a time. Not thread-safe: call it from a single thread (RequestQueue).
//
// generate(): reuse cached prefix -> chunked prefill of the rest -> decode loop
//             (graph replay, sample on GPU, copy 1 token back, detokenize, stop check).
class Engine {
 public:
  explicit Engine(const EngineConfig& config);
  ~Engine();

  // Throws std::invalid_argument if the prompt is empty or doesn't fit in the context.
  // max_tokens is clamped so prompt + completion fits.
  GenerationResult generate(const GenerationRequest& request, const TokenCallback& on_token);

  const Tokenizer& tokenizer() const;
  const ChatTemplate& chat_template() const;
  const std::string& model_name() const;  // the model directory's name
  int max_seq_len() const;                // KV cache capacity in tokens

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ography
