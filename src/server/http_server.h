#pragma once

#include <memory>
#include <string>

namespace ography {

class Engine;

struct ServerConfig {
  std::string host = "127.0.0.1";  // localhost only: private, single-user
  int port = 8000;
  std::string api_key;             // if set, require "Authorization: Bearer <api_key>"
};

// OpenAI-compatible HTTP server.
//   POST /v1/chat/completions   (stream and non-stream)
//   POST /v1/completions
//   GET  /v1/models
//   GET  /health
// Requests are forwarded to the Engine through a RequestQueue; streaming responses
// use chunked transfer + SSE, and a client disconnect cancels generation.
class HttpServer {
 public:
  HttpServer(Engine& engine, ServerConfig config);
  ~HttpServer();

  void run();   // blocks until stop()
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ography
