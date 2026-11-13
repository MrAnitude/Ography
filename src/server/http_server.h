#pragma once

#include <memory>
#include <string>

namespace ography {

class Engine;

struct ServerConfig {
  std::string host = "127.0.0.1";  // localhost only: private, single-user
  int port = 8000;                 // 0 => any free port (see bind())
  std::string api_key;             // if set, require "Authorization: Bearer <api_key>" on /v1/*
};

// OpenAI-compatible HTTP server.
//   POST /v1/chat/completions   (stream and non-stream)
//   POST /v1/completions        (stream and non-stream)
//   GET  /v1/models
//   GET  /health                (status plus request counters)
// Prompts are tokenized on the HTTP thread (so bad requests fail with a 400 up front), then
// generation runs on the RequestQueue's single GPU thread. Streaming responses use chunked
// transfer + SSE, and a client disconnect cancels generation. A non-streaming request whose
// client disconnects runs to completion (bounded by max_tokens): httplib can't tell earlier.
class HttpServer {
 public:
  HttpServer(Engine& engine, ServerConfig config);
  ~HttpServer();

  // Binds and starts listening (connections queue until run()); returns the port.
  // Throws if the address can't be bound. run() calls it if needed.
  int bind();
  void run();   // serves until stop()
  void stop();  // thread-safe

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ography
