#include "server/openai_api.h"

#include <nlohmann/json.hpp>

#include "core/logging.h"

namespace ography::openai {

ChatCompletionRequest parse_chat_request(const nlohmann::json& body) {
  OGRAPHY_TODO("openai::parse_chat_request");
}

CompletionRequest parse_completion_request(const nlohmann::json& body) {
  OGRAPHY_TODO("openai::parse_completion_request");
}

nlohmann::json chat_response(const std::string& id, const std::string& model,
                             const std::string& content, const GenerationResult& result) {
  OGRAPHY_TODO("openai::chat_response");
}

nlohmann::json completion_response(const std::string& id, const std::string& model,
                                   const std::string& text, const GenerationResult& result) {
  OGRAPHY_TODO("openai::completion_response");
}

nlohmann::json chat_chunk(const std::string& id, const std::string& model,
                          const std::string& delta, std::optional<FinishReason> finish) {
  OGRAPHY_TODO("openai::chat_chunk");
}

nlohmann::json error_body(const std::string& message, const std::string& type) {
  return {{"error", {{"message", message}, {"type", type}}}};
}

std::string finish_reason_name(FinishReason reason) {
  switch (reason) {
    case FinishReason::Stop: return "stop";
    case FinishReason::Length: return "length";
    case FinishReason::Cancelled: return "stop";
  }
  return "stop";
}

}  // namespace ography::openai
