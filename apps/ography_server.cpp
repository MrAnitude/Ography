// OpenAI-compatible inference server.
//
//   ography_server --model /path/to/hf-model [--host 127.0.0.1] [--port 8000]
//                  [--max-seq-len N] [--fp8-kv] [--no-cuda-graphs] [--api-key KEY]

#include "core/logging.h"
#include "engine/engine.h"
#include "server/http_server.h"

int main(int argc, char** argv) {
  using namespace ography;
  // TODO: parse arguments into EngineConfig / ServerConfig.
  EngineConfig engine_config;
  ServerConfig server_config;

  Engine engine(engine_config);
  HttpServer server(engine, server_config);
  log(LogLevel::Info, "listening");
  server.run();
  return 0;
}
