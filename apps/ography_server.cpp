// OpenAI-compatible inference server.
//
//   ography_server --model /path/to/hf-model [--host 127.0.0.1] [--port 8000]
//                  [--max-seq-len N] [--prefill-chunk N] [--api-key KEY] [--no-cuda-graphs] [--fp8-kv]
//
// Ctrl-C / SIGTERM stops the server; generations already queued finish first.

#include <pthread.h>
#include <signal.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <thread>

#include "core/logging.h"
#include "engine/engine.h"
#include "server/http_server.h"

namespace {

using namespace ography;

[[noreturn]] void usage(const char* argv0, int code) {
  std::fprintf(code ? stderr : stdout,
               "usage: %s --model DIR [options]\n"
               "  --host ADDR          bind address, default 127.0.0.1 (local only)\n"
               "  --port N             default 8000\n"
               "  --api-key KEY        require \"Authorization: Bearer KEY\" on /v1/*\n"
               "                       (default: $OGRAPHY_API_KEY, else none)\n"
               "  --max-seq-len N      context length, default as large as fits\n"
               "  --prefill-chunk N    default 2048\n"
               "  --vram-fraction F    share of free VRAM to claim, default 0.92\n"
               "  --no-cuda-graphs     decode without graph replay (debugging)\n"
               "  --fp8-kv             store the KV cache in FP8: twice the context\n",
               argv0);
  std::exit(code);
}

}  // namespace

int main(int argc, char** argv) {
  EngineConfig engine_config;
  ServerConfig server_config;
  if (const char* key = std::getenv("OGRAPHY_API_KEY")) server_config.api_key = key;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage(argv[0], 2);
      return argv[++i];
    };
    if (a == "--model") engine_config.model_dir = next();
    else if (a == "--host") server_config.host = next();
    else if (a == "--port") server_config.port = std::stoi(next());
    else if (a == "--api-key") server_config.api_key = next();
    else if (a == "--max-seq-len") engine_config.max_seq_len = std::stoi(next());
    else if (a == "--prefill-chunk") engine_config.prefill_chunk = std::stoi(next());
    else if (a == "--vram-fraction") engine_config.vram_fraction = std::stof(next());
    else if (a == "--no-cuda-graphs") engine_config.use_cuda_graphs = false;
    else if (a == "--fp8-kv") engine_config.fp8_kv_cache = true;
    else if (a == "-h" || a == "--help") usage(argv[0], 0);
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage(argv[0], 2);
    }
  }
  if (engine_config.model_dir.empty()) usage(argv[0], 2);

  // Block SIGINT/SIGTERM in every thread (they inherit the mask) and handle them on one
  // dedicated thread with sigwait, so stopping the server never runs in a signal handler.
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGINT);
  sigaddset(&signals, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);

  try {
    Engine engine(engine_config);
    HttpServer server(engine, server_config);
    const int port = server.bind();
    if (server_config.host != "127.0.0.1" && server_config.host != "localhost" &&
        server_config.api_key.empty()) {
      log(LogLevel::Warn, "listening beyond localhost without --api-key");
    }

    std::thread signal_thread([&] {
      int sig = 0;
      sigwait(&signals, &sig);
      log(LogLevel::Info, "shutting down");
      server.stop();
    });

    log(LogLevel::Info, "serving " + engine.model_name() + " on http://" + server_config.host +
                            ":" + std::to_string(port) + "/v1");
    server.run();

    // run() also returns if listening fails; make sure the signal thread can exit.
    pthread_kill(signal_thread.native_handle(), SIGTERM);
    signal_thread.join();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
