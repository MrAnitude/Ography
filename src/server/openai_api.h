#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "engine/engine.h"
#include "tokenizer/chat_template.h"

namespace ography::openai {

// POST /v1/chat/completions
struct ChatCompletionRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  SamplingParams sampling;  // temperature, top_p, max_tokens, stop, seed, penalties
  bool stream = false;
  bool include_usage = false;  // stream_options.include_usage
};

// POST /v1/completions
struct CompletionRequest {
  std::string model;
  std::string prompt;
  SamplingParams sampling;
  bool stream = false;
  bool include_usage = false;
};

// Throws std::invalid_argument with a client-facing message on bad input. Unsupported
// features (n > 1, tools, logprobs, images, ...) are rejected rather than silently ignored.
// Besides the OpenAI fields, top_k, min_p and repetition_penalty are accepted (as in vLLM).
// max_tokens defaults to "as many as fit in the context".
ChatCompletionRequest parse_chat_request(const nlohmann::json& body);
CompletionRequest parse_completion_request(const nlohmann::json& body);

// Response metadata shared by every chunk of one response.
struct ResponseMeta {
  std::string id;     // "chatcmpl-..." / "cmpl-..."
  std::string model;
  int64_t created = 0;  // unix seconds
};
ResponseMeta new_meta(const std::string& prefix, const std::string& model);

// Full (non-streaming) response bodies.
nlohmann::json chat_response(const ResponseMeta& meta, const std::string& content,
                             const GenerationResult& result);
nlohmann::json completion_response(const ResponseMeta& meta, const std::string& text,
                                   const GenerationResult& result);

// Streaming chunks. A chat stream starts with a {"role": "assistant"} delta, then one chunk per
// text piece, then an empty delta carrying finish_reason, then (if requested) a usage chunk.
nlohmann::json chat_chunk(const ResponseMeta& meta, const nlohmann::json& delta,
                          std::optional<FinishReason> finish);
nlohmann::json completion_chunk(const ResponseMeta& meta, const std::string& text,
                                std::optional<FinishReason> finish);
nlohmann::json usage_chunk(const ResponseMeta& meta, const char* object,
                           const GenerationResult& result);

nlohmann::json usage(const GenerationResult& result);
nlohmann::json error_body(const std::string& message, const std::string& type,
                          const std::string& code = "");
std::string finish_reason_name(FinishReason reason);

}  // namespace ography::openai
