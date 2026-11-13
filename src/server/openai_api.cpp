#include "server/openai_api.h"

#include <chrono>
#include <climits>
#include <random>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace ography::openai {

using nlohmann::json;

namespace {

[[noreturn]] void bad(const std::string& msg) { throw std::invalid_argument(msg); }

float number(const json& body, const char* key, float def, float lo, float hi) {
  const auto it = body.find(key);
  if (it == body.end() || it->is_null()) return def;
  if (!it->is_number()) bad(std::string("'") + key + "' must be a number");
  const float v = it->get<float>();
  if (v < lo || v > hi) {
    bad(std::string("'") + key + "' must be between " + std::to_string(lo) + " and " +
        std::to_string(hi));
  }
  return v;
}

int integer(const json& body, const char* key, int def, int lo) {
  const auto it = body.find(key);
  if (it == body.end() || it->is_null()) return def;
  if (!it->is_number_integer()) bad(std::string("'") + key + "' must be an integer");
  const int64_t v = it->get<int64_t>();
  if (v < lo) bad(std::string("'") + key + "' must be at least " + std::to_string(lo));
  return static_cast<int>(std::min<int64_t>(v, INT_MAX));
}

bool boolean(const json& body, const char* key) {
  const auto it = body.find(key);
  if (it == body.end() || it->is_null()) return false;
  if (!it->is_boolean()) bad(std::string("'") + key + "' must be a boolean");
  return it->get<bool>();
}

// Fields common to both endpoints.
void parse_common(const json& body, SamplingParams& sp, std::string& model, bool& stream,
                  bool& include_usage) {
  if (!body.is_object()) bad("request body must be a JSON object");
  if (body.contains("model") && !body["model"].is_null()) {
    if (!body["model"].is_string()) bad("'model' must be a string");
    model = body["model"].get<std::string>();  // any name is accepted: one model is served
  }
  if (integer(body, "n", 1, 1) != 1) bad("only n = 1 is supported");
  for (const char* key : {"tools", "functions", "tool_choice", "function_call"}) {
    if (body.contains(key) && !body[key].is_null()) bad(std::string("'") + key + "' is not supported");
  }
  if (boolean(body, "logprobs") || (body.contains("top_logprobs") && !body["top_logprobs"].is_null())) {
    bad("logprobs are not supported");
  }
  if (body.contains("response_format") && body["response_format"].is_object() &&
      body["response_format"].value("type", "text") != "text") {
    bad("response_format other than text is not supported");
  }

  sp.temperature = number(body, "temperature", 1.0f, 0.0f, 2.0f);
  sp.top_p = number(body, "top_p", 1.0f, 0.0f, 1.0f);
  if (sp.top_p == 0.0f) sp.temperature = 0.0f;  // top_p 0 can only mean "the top token"
  sp.top_k = integer(body, "top_k", 0, 0);
  sp.min_p = number(body, "min_p", 0.0f, 0.0f, 1.0f);
  sp.presence_penalty = number(body, "presence_penalty", 0.0f, -2.0f, 2.0f);
  sp.frequency_penalty = number(body, "frequency_penalty", 0.0f, -2.0f, 2.0f);
  sp.repetition_penalty = number(body, "repetition_penalty", 1.0f, 0.01f, 10.0f);
  // max_completion_tokens is the newer name; the engine clamps to the context.
  sp.max_tokens = integer(body, "max_completion_tokens", integer(body, "max_tokens", INT_MAX, 1), 1);
  if (body.contains("seed") && !body["seed"].is_null()) {
    if (!body["seed"].is_number_integer()) bad("'seed' must be an integer");
    sp.seed = static_cast<uint64_t>(body["seed"].get<int64_t>());
    if (sp.seed == 0) sp.seed = 0x5EED;  // 0 means "random" internally
  }

  if (body.contains("stop") && !body["stop"].is_null()) {
    const json& stop = body["stop"];
    if (stop.is_string()) {
      sp.stop = {stop.get<std::string>()};
    } else if (stop.is_array()) {
      if (stop.size() > 4) bad("at most 4 stop sequences are allowed");
      for (const auto& s : stop) {
        if (!s.is_string()) bad("'stop' must be a string or an array of strings");
        sp.stop.push_back(s.get<std::string>());
      }
    } else {
      bad("'stop' must be a string or an array of strings");
    }
  }

  stream = boolean(body, "stream");
  if (body.contains("stream_options") && body["stream_options"].is_object()) {
    include_usage = body["stream_options"].value("include_usage", false);
  }
}

// Message content: a string, or an array of {"type": "text", "text": ...} parts.
std::string message_content(const json& content) {
  if (content.is_null()) return "";
  if (content.is_string()) return content.get<std::string>();
  if (content.is_array()) {
    std::string out;
    for (const auto& part : content) {
      if (!part.is_object() || part.value("type", "") != "text" || !part.contains("text")) {
        bad("only text content parts are supported");
      }
      out += part["text"].get<std::string>();
    }
    return out;
  }
  bad("message 'content' must be a string or an array of text parts");
}

}  // namespace

ChatCompletionRequest parse_chat_request(const json& body) {
  ChatCompletionRequest r;
  parse_common(body, r.sampling, r.model, r.stream, r.include_usage);
  const auto it = body.find("messages");
  if (it == body.end() || !it->is_array() || it->empty()) {
    bad("'messages' must be a non-empty array");
  }
  for (const auto& m : *it) {
    if (!m.is_object() || !m.contains("role") || !m["role"].is_string()) {
      bad("each message needs a string 'role'");
    }
    std::string role = m["role"].get<std::string>();
    if (role == "developer") role = "system";  // OpenAI's newer name for system
    if (role != "system" && role != "user" && role != "assistant") {
      bad("unsupported message role '" + role + "'");
    }
    r.messages.push_back({role, message_content(m.value("content", json()))});
  }
  return r;
}

CompletionRequest parse_completion_request(const json& body) {
  CompletionRequest r;
  parse_common(body, r.sampling, r.model, r.stream, r.include_usage);
  if (r.sampling.max_tokens == INT_MAX && !body.contains("max_tokens")) {
    r.sampling.max_tokens = 16;  // the OpenAI default for completions
  }
  const auto it = body.find("prompt");
  if (it == body.end()) bad("'prompt' is required");
  if (it->is_string()) {
    r.prompt = it->get<std::string>();
  } else if (it->is_array() && it->size() == 1 && (*it)[0].is_string()) {
    r.prompt = (*it)[0].get<std::string>();
  } else {
    bad("'prompt' must be a string (batches of prompts are not supported)");
  }
  if (r.prompt.empty()) bad("'prompt' must not be empty");
  return r;
}

ResponseMeta new_meta(const std::string& prefix, const std::string& model) {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  static const char* hex = "0123456789abcdef";
  std::string id = prefix + "-";
  for (int i = 0; i < 24; ++i) id += hex[rng() % 16];
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return {id, model, std::chrono::duration_cast<std::chrono::seconds>(now).count()};
}

json usage(const GenerationResult& r) {
  json u = {{"prompt_tokens", r.prompt_tokens},
            {"completion_tokens", r.completion_tokens},
            {"total_tokens", r.prompt_tokens + r.completion_tokens}};
  u["prompt_tokens_details"] = {{"cached_tokens", r.cached_prompt_tokens}};
  return u;
}

json chat_response(const ResponseMeta& meta, const std::string& content,
                   const GenerationResult& result) {
  return {{"id", meta.id},
          {"object", "chat.completion"},
          {"created", meta.created},
          {"model", meta.model},
          {"choices", json::array({{{"index", 0},
                                    {"message", {{"role", "assistant"}, {"content", content}}},
                                    {"logprobs", nullptr},
                                    {"finish_reason", finish_reason_name(result.finish_reason)}}})},
          {"usage", usage(result)}};
}

json completion_response(const ResponseMeta& meta, const std::string& text,
                         const GenerationResult& result) {
  return {{"id", meta.id},
          {"object", "text_completion"},
          {"created", meta.created},
          {"model", meta.model},
          {"choices", json::array({{{"index", 0},
                                    {"text", text},
                                    {"logprobs", nullptr},
                                    {"finish_reason", finish_reason_name(result.finish_reason)}}})},
          {"usage", usage(result)}};
}

json chat_chunk(const ResponseMeta& meta, const json& delta, std::optional<FinishReason> finish) {
  return {{"id", meta.id},
          {"object", "chat.completion.chunk"},
          {"created", meta.created},
          {"model", meta.model},
          {"choices", json::array({{{"index", 0},
                                    {"delta", delta},
                                    {"logprobs", nullptr},
                                    {"finish_reason", finish ? json(finish_reason_name(*finish))
                                                             : json(nullptr)}}})}};
}

json completion_chunk(const ResponseMeta& meta, const std::string& text,
                      std::optional<FinishReason> finish) {
  return {{"id", meta.id},
          {"object", "text_completion"},
          {"created", meta.created},
          {"model", meta.model},
          {"choices", json::array({{{"index", 0},
                                    {"text", text},
                                    {"logprobs", nullptr},
                                    {"finish_reason", finish ? json(finish_reason_name(*finish))
                                                             : json(nullptr)}}})}};
}

json usage_chunk(const ResponseMeta& meta, const char* object, const GenerationResult& result) {
  return {{"id", meta.id},       {"object", object},          {"created", meta.created},
          {"model", meta.model}, {"choices", json::array()}, {"usage", usage(result)}};
}

json error_body(const std::string& message, const std::string& type, const std::string& code) {
  return {{"error",
           {{"message", message}, {"type", type}, {"param", nullptr},
            {"code", code.empty() ? json(nullptr) : json(code)}}}};
}

std::string finish_reason_name(FinishReason reason) {
  switch (reason) {
    case FinishReason::Stop: return "stop";
    case FinishReason::Length: return "length";
    case FinishReason::Cancelled: return "stop";  // only seen by a client that went away
  }
  return "stop";
}

}  // namespace ography::openai
