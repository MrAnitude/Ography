#include "server/http_server.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "core/logging.h"
#include "engine/engine.h"
#include "engine/request_queue.h"
#include "server/openai_api.h"
#include "server/sse.h"

namespace ography {

struct HttpServer::Impl {
  Engine& engine;
  ServerConfig config;
  httplib::Server http;

  Impl(Engine& e, ServerConfig c) : engine(e), config(std::move(c)) {}
  // TODO: std::unique_ptr<RequestQueue> queue;
};

HttpServer::HttpServer(Engine& engine, ServerConfig config)
    : impl_(std::make_unique<Impl>(engine, std::move(config))) {
  // TODO: register routes, auth check, error handler.
}

HttpServer::~HttpServer() = default;

void HttpServer::run() { OGRAPHY_TODO("HttpServer::run"); }

void HttpServer::stop() { impl_->http.stop(); }

}  // namespace ography
