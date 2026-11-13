#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "server/sse.h"

using namespace ography;

TEST(Sse, EventFraming) {
  EXPECT_EQ(sse::event(nlohmann::json{{"a", 1}}), "data: {\"a\":1}\n\n");
  EXPECT_EQ(sse::done(), "data: [DONE]\n\n");
}

// TODO: parse_chat_request maps OpenAI fields to SamplingParams; rejects bad input.
TEST(OpenAI, ParseChatRequest) { GTEST_SKIP() << "not implemented yet"; }
