// The HTTP server end to end on GPT-2: a real HttpServer on a free port, driven with httplib's
// client. Skipped if the checkpoint isn't present.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "engine/engine.h"
#include "server/http_server.h"

using namespace ography;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

std::string gpt2_dir() {
  if (const char* env = std::getenv("OGRAPHY_GPT2_DIR")) return env;
  return std::string(OGRAPHY_SOURCE_DIR) + "/models/gpt2";
}

constexpr const char* kKey = "test-key";

// Parses "data: ..." SSE events; [DONE] becomes the string "[DONE]".
std::vector<json> sse_events(const std::string& body) {
  std::vector<json> out;
  size_t pos = 0;
  while ((pos = body.find("data: ", pos)) != std::string::npos) {
    const size_t end = body.find("\n\n", pos);
    const std::string data = body.substr(pos + 6, end - pos - 6);
    out.push_back(data == "[DONE]" ? json("[DONE]") : json::parse(data));
    pos = end;
  }
  return out;
}

class ServerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!fs::exists(gpt2_dir() + "/model.safetensors")) return;
    EngineConfig ec;
    ec.model_dir = gpt2_dir();
    engine_ = std::make_unique<Engine>(ec);
    ServerConfig sc;
    sc.port = 0;
    sc.api_key = kKey;
    server_ = std::make_unique<HttpServer>(*engine_, sc);
    port_ = server_->bind();
    thread_ = std::thread([] { server_->run(); });
  }
  static void TearDownTestSuite() {
    if (server_) server_->stop();
    if (thread_.joinable()) thread_.join();
    server_.reset();
    engine_.reset();
  }
  void SetUp() override {
    if (!engine_) GTEST_SKIP() << "GPT-2 checkpoint not found at " << gpt2_dir();
  }

  static httplib::Client client(bool auth = true) {
    httplib::Client c("127.0.0.1", port_);
    c.set_read_timeout(60, 0);
    if (auth) c.set_bearer_token_auth(kKey);
    return c;
  }
  static json post(const std::string& path, const json& body, int expect_status = 200) {
    auto c = client();
    auto res = c.Post(path, body.dump(), "application/json");
    EXPECT_TRUE(res);
    if (!res) return {};
    EXPECT_EQ(res->status, expect_status) << res->body;
    return json::parse(res->body);
  }
  static json health() {
    auto c = client();
    return json::parse(c.Get("/health")->body);
  }

  static std::unique_ptr<Engine> engine_;
  static std::unique_ptr<HttpServer> server_;
  static std::thread thread_;
  static int port_;
};

std::unique_ptr<Engine> ServerTest::engine_;
std::unique_ptr<HttpServer> ServerTest::server_;
std::thread ServerTest::thread_;
int ServerTest::port_ = 0;

}  // namespace

TEST_F(ServerTest, HealthAndModels) {
  EXPECT_EQ(health()["status"], "ok");
  auto c = client();
  const json models = json::parse(c.Get("/v1/models")->body);
  EXPECT_EQ(models["data"][0]["id"], "gpt2");
  EXPECT_EQ(models["data"][0]["max_model_len"], 1024);
}

TEST_F(ServerTest, RequiresApiKey) {
  auto c = client(/*auth=*/false);
  auto res = c.Get("/v1/models");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, 401);
  EXPECT_EQ(json::parse(res->body)["error"]["code"], "invalid_api_key");
  EXPECT_EQ(c.Get("/health")->status, 200);  // health stays open
}

TEST_F(ServerTest, CompletionMatchesHFGreedy) {
  const json r = post("/v1/completions",
                      {{"prompt", "The capital of France is"}, {"max_tokens", 8}, {"temperature", 0}});
  EXPECT_EQ(r["object"], "text_completion");
  EXPECT_EQ(r["choices"][0]["text"], " the capital of the French Republic, and");
  EXPECT_EQ(r["choices"][0]["finish_reason"], "length");
  EXPECT_EQ(r["usage"]["prompt_tokens"], 5);
  EXPECT_EQ(r["usage"]["completion_tokens"], 8);
}

TEST_F(ServerTest, StreamingCompletion) {
  auto c = client();
  auto res = c.Post("/v1/completions",
                    json{{"prompt", "The capital of France is"}, {"max_tokens", 8},
                         {"temperature", 0}, {"stream", true},
                         {"stream_options", {{"include_usage", true}}}}
                        .dump(),
                    "application/json");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->get_header_value("Content-Type"), "text/event-stream");
  const auto events = sse_events(res->body);
  ASSERT_GE(events.size(), 4u);
  std::string text;
  for (size_t i = 0; i + 3 < events.size(); ++i) text += events[i]["choices"][0]["text"];
  EXPECT_EQ(text, " the capital of the French Republic, and");
  EXPECT_EQ(events[events.size() - 3]["choices"][0]["finish_reason"], "length");
  EXPECT_EQ(events[events.size() - 2]["usage"]["completion_tokens"], 8);
  EXPECT_EQ(events.back(), "[DONE]");
}

TEST_F(ServerTest, StopSequence) {
  const json r = post("/v1/completions", {{"prompt", "The capital of France is"},
                                          {"max_tokens", 20}, {"temperature", 0},
                                          {"stop", " French"}});
  EXPECT_EQ(r["choices"][0]["text"], " the capital of the");
  EXPECT_EQ(r["choices"][0]["finish_reason"], "stop");
}

TEST_F(ServerTest, ChatNeedsATemplate) {
  const json r = post("/v1/chat/completions", {{"messages", {{{"role", "user"}, {"content", "hi"}}}}},
                      400);
  EXPECT_NE(r["error"]["message"].get<std::string>().find("no chat template"), std::string::npos);
}

TEST_F(ServerTest, BadRequests) {
  auto c = client();
  auto res = c.Post("/v1/completions", "{not json", "application/json");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, 400);
  EXPECT_EQ(json::parse(res->body)["error"]["type"], "invalid_request_error");

  std::string long_prompt;
  for (int i = 0; i < 1100; ++i) long_prompt += " the";
  const json r = post("/v1/completions", {{"prompt", long_prompt}}, 400);
  EXPECT_EQ(r["error"]["code"], "context_length_exceeded");

  res = c.Get("/v1/nothing");
  ASSERT_TRUE(res);
  EXPECT_EQ(res->status, 404);
  EXPECT_TRUE(json::parse(res->body).contains("error"));
}

TEST_F(ServerTest, ClientDisconnectCancelsGeneration) {
  const int64_t before = health()["cancelled"];
  // Read a few events of a long stream, then hang up.
  httplib::Request req;
  req.method = "POST";
  req.path = "/v1/completions";
  req.set_header("Authorization", std::string("Bearer ") + kKey);
  req.set_header("Content-Type", "application/json");
  req.body = json{{"prompt", "Once upon a time"}, {"max_tokens", 1000}, {"stream", true},
                  {"temperature", 0.8}, {"seed", 1}}
                 .dump();
  int received = 0;
  req.content_receiver = [&](const char*, size_t, uint64_t, uint64_t) { return ++received < 3; };
  auto c = client();
  const auto t0 = std::chrono::steady_clock::now();
  c.send(req);
  EXPECT_EQ(received, 3);

  // The engine must stop well before 1000 tokens (~1.5 s at GPT-2 speed).
  int64_t after = before;
  while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
    const json h = health();
    after = h["cancelled"];
    if (after > before && h["in_flight"] == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  EXPECT_EQ(after, before + 1);
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::milliseconds(1000));

  // And the server keeps serving.
  EXPECT_EQ(post("/v1/completions", {{"prompt", "Hi"}, {"max_tokens", 2}})["usage"]["completion_tokens"], 2);
}
