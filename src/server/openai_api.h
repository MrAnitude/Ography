#pragma once

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
};

// POST /v1/completions
struct CompletionRequest {
  std::string model;
  std::string prompt;
  SamplingParams sampling;
  bool stream = false;
};

// Throws std::invalid_argument with a client-facing message on bad input.
ChatCompletionRequest parse_chat_request(const nlohmann::json& body);
CompletionRequest parse_completion_request(const nlohmann::json& body);

// Full (non-streaming) response bodies.
nlohmann::json chat_response(const std::string& id, const std::string& model,
                             const std::string& content, const GenerationResult& result);
nlohmann::json completion_response(const std::string& id, const std::string& model,
                                   const std::string& text, const GenerationResult& result);

// One streaming chunk ("chat.completion.chunk"). finish set only on the final chunk.
nlohmann::json chat_chunk(const std::string& id, const std::string& model,
                          const std::string& delta, std::optional<FinishReason> finish);

nlohmann::json error_body(const std::string& message, const std::string& type);
std::string finish_reason_name(FinishReason reason);

}  // namespace ography::openai
