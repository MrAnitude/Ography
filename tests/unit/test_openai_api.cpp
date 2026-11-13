#include <gtest/gtest.h>

#include <climits>

#include <nlohmann/json.hpp>

#include "server/openai_api.h"
#include "server/sse.h"

using namespace ography;
using nlohmann::json;

TEST(Sse, EventFraming) {
  EXPECT_EQ(sse::event(json{{"a", 1}}), "data: {\"a\":1}\n\n");
  EXPECT_EQ(sse::done(), "data: [DONE]\n\n");
}

TEST(OpenAI, ParseChatRequest) {
  const auto r = openai::parse_chat_request(json::parse(R"({
    "model": "anything", "stream": true, "stream_options": {"include_usage": true},
    "temperature": 0.3, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "max_tokens": 77,
    "presence_penalty": 0.5, "frequency_penalty": -0.25, "repetition_penalty": 1.1,
    "seed": 42, "stop": ["\n\n", "END"],
    "messages": [
      {"role": "developer", "content": "Be terse."},
      {"role": "user", "content": [{"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}]},
      {"role": "assistant", "content": null}
    ]})"));
  EXPECT_EQ(r.model, "anything");
  EXPECT_TRUE(r.stream);
  EXPECT_TRUE(r.include_usage);
  ASSERT_EQ(r.messages.size(), 3u);
  EXPECT_EQ(r.messages[0].role, "system");  // developer -> system
  EXPECT_EQ(r.messages[1].content, "Hello");
  EXPECT_EQ(r.messages[2].content, "");
  const SamplingParams& s = r.sampling;
  EXPECT_FLOAT_EQ(s.temperature, 0.3f);
  EXPECT_FLOAT_EQ(s.top_p, 0.9f);
  EXPECT_EQ(s.top_k, 40);
  EXPECT_FLOAT_EQ(s.min_p, 0.05f);
  EXPECT_EQ(s.max_tokens, 77);
  EXPECT_FLOAT_EQ(s.presence_penalty, 0.5f);
  EXPECT_FLOAT_EQ(s.frequency_penalty, -0.25f);
  EXPECT_FLOAT_EQ(s.repetition_penalty, 1.1f);
  EXPECT_EQ(s.seed, 42u);
  EXPECT_EQ(s.stop, (std::vector<std::string>{"\n\n", "END"}));
}

TEST(OpenAI, ChatDefaults) {
  const auto r = openai::parse_chat_request(
      json::parse(R"({"messages": [{"role": "user", "content": "hi"}], "stop": "x"})"));
  EXPECT_FALSE(r.stream);
  EXPECT_FLOAT_EQ(r.sampling.temperature, 1.0f);
  EXPECT_EQ(r.sampling.max_tokens, INT_MAX);  // the engine clamps to the context
  EXPECT_EQ(r.sampling.seed, 0u);             // random
  EXPECT_EQ(r.sampling.stop, std::vector<std::string>{"x"});
  // max_completion_tokens wins over max_tokens.
  EXPECT_EQ(openai::parse_chat_request(json::parse(
                R"({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5,
                    "max_completion_tokens": 9})"))
                .sampling.max_tokens,
            9);
}

TEST(OpenAI, RejectsBadOrUnsupportedRequests) {
  const char* bad[] = {
      R"([])",
      R"({})",
      R"({"messages": []})",
      R"({"messages": [{"role": "tool", "content": "x"}]})",
      R"({"messages": [{"content": "x"}]})",
      R"({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})",
      R"({"messages": [{"role": "user", "content": "x"}], "n": 2})",
      R"({"messages": [{"role": "user", "content": "x"}], "tools": []})",
      R"({"messages": [{"role": "user", "content": "x"}], "logprobs": true})",
      R"({"messages": [{"role": "user", "content": "x"}], "temperature": 3})",
      R"({"messages": [{"role": "user", "content": "x"}], "temperature": "hot"})",
      R"({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0})",
      R"({"messages": [{"role": "user", "content": "x"}], "stop": ["a","b","c","d","e"]})",
      R"({"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}})",
  };
  for (const char* body : bad) {
    EXPECT_THROW(openai::parse_chat_request(json::parse(body)), std::invalid_argument) << body;
  }
}

TEST(OpenAI, ParseCompletionRequest) {
  auto r = openai::parse_completion_request(json::parse(R"({"prompt": "Once upon"})"));
  EXPECT_EQ(r.prompt, "Once upon");
  EXPECT_EQ(r.sampling.max_tokens, 16);  // OpenAI's completions default
  r = openai::parse_completion_request(json::parse(R"({"prompt": ["x"], "max_tokens": 3})"));
  EXPECT_EQ(r.prompt, "x");
  EXPECT_EQ(r.sampling.max_tokens, 3);
  EXPECT_THROW(openai::parse_completion_request(json::parse(R"({"prompt": ["a", "b"]})")),
               std::invalid_argument);
  EXPECT_THROW(openai::parse_completion_request(json::parse(R"({"prompt": ""})")),
               std::invalid_argument);
}

TEST(OpenAI, ResponseShapes) {
  const auto meta = openai::new_meta("chatcmpl", "m");
  EXPECT_EQ(meta.id.rfind("chatcmpl-", 0), 0u);
  GenerationResult res;
  res.prompt_tokens = 10;
  res.completion_tokens = 5;
  res.finish_reason = FinishReason::Length;

  const json full = openai::chat_response(meta, "hi", res);
  EXPECT_EQ(full["object"], "chat.completion");
  EXPECT_EQ(full["choices"][0]["message"]["content"], "hi");
  EXPECT_EQ(full["choices"][0]["finish_reason"], "length");
  EXPECT_EQ(full["usage"]["total_tokens"], 15);

  const json chunk = openai::chat_chunk(meta, json{{"content", "x"}}, std::nullopt);
  EXPECT_EQ(chunk["object"], "chat.completion.chunk");
  EXPECT_TRUE(chunk["choices"][0]["finish_reason"].is_null());
  const json last = openai::chat_chunk(meta, json::object(), FinishReason::Stop);
  EXPECT_EQ(last["choices"][0]["finish_reason"], "stop");
  EXPECT_TRUE(openai::usage_chunk(meta, "chat.completion.chunk", res)["choices"].empty());

  EXPECT_EQ(openai::completion_response(meta, "t", res)["choices"][0]["text"], "t");
  EXPECT_EQ(openai::error_body("m", "invalid_request_error", "c")["error"]["code"], "c");
}
