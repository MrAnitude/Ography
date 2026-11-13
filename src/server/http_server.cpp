#include "server/http_server.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <stdexcept>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "core/logging.h"
#include "engine/engine.h"
#include "engine/request_queue.h"
#include "server/openai_api.h"
#include "server/sse.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"

namespace ography {

using nlohmann::json;

namespace {

// Hands text from the GPU thread (producer) to the HTTP thread (consumer).
struct Channel {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::string> pieces;
  bool finished = false;
  std::string error;  // set if generate() threw
  GenerationResult result;
  std::atomic<bool> cancelled{false};  // set by the HTTP side when the client goes away

  void push(std::string piece) {
    {
      std::lock_guard<std::mutex> lock(mu);
      pieces.push_back(std::move(piece));
    }
    cv.notify_one();
  }
  void finish(GenerationResult r, std::string err = "") {
    {
      std::lock_guard<std::mutex> lock(mu);
      result = r;
      error = std::move(err);
      finished = true;
    }
    cv.notify_one();
  }
  // Waits for text or the end; moves out any pending pieces. Returns false once finished
  // and drained.
  bool next(std::vector<std::string>& out) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return finished || !pieces.empty(); });
    out.assign(std::make_move_iterator(pieces.begin()), std::make_move_iterator(pieces.end()));
    pieces.clear();
    return !(finished && out.empty());
  }
};

void send_json(httplib::Response& res, int status, const json& body) {
  res.status = status;
  res.set_content(body.dump(), "application/json");
}

void send_error(httplib::Response& res, int status, const std::string& message,
                const std::string& type, const std::string& code = "") {
  send_json(res, status, openai::error_body(message, type, code));
}

}  // namespace

struct HttpServer::Impl {
  Engine& engine;
  ServerConfig config;
  httplib::Server http;
  RequestQueue queue;
  bool bound = false;
  int port = 0;

  std::atomic<int64_t> requests{0}, cancelled{0}, in_flight{0};

  Impl(Engine& e, ServerConfig c) : engine(e), config(std::move(c)), queue(e) {}

  // Queues a generation; text and the result arrive on the returned channel.
  std::shared_ptr<Channel> start(std::vector<int32_t> prompt, SamplingParams sp,
                                 const std::string& what) {
    auto ch = std::make_shared<Channel>();
    ++requests;
    ++in_flight;
    queue.submit([this, ch, prompt = std::move(prompt), sp, what](Engine& engine) {
      GenerationResult r;
      r.prompt_tokens = static_cast<int>(prompt.size());
      r.finish_reason = FinishReason::Cancelled;
      try {
        if (!ch->cancelled) {  // the client may have left while queued
          r = engine.generate({prompt, sp}, [&](const TokenEvent& ev) {
            if (!ev.text.empty()) ch->push(ev.text);
            return !ch->cancelled.load();
          });
        }
        ch->finish(r);
      } catch (const std::exception& e) {
        ch->finish(r, e.what());
      }
      --in_flight;
      if (r.finish_reason == FinishReason::Cancelled) ++cancelled;
      char msg[256];
      std::snprintf(msg, sizeof msg, "%s: %d+%d tokens, TTFT %.0f ms, %.1f tok/s, %s",
                    what.c_str(), r.prompt_tokens, r.completion_tokens, r.time_to_first_token_ms,
                    r.tokens_per_second,
                    r.finish_reason == FinishReason::Cancelled
                        ? "cancelled"
                        : openai::finish_reason_name(r.finish_reason).c_str());
      log(LogLevel::Info, msg);
    });
    return ch;
  }

  // Rejects prompts that can't fit before anything is queued or streamed.
  bool check_length(const std::vector<int32_t>& prompt, httplib::Response& res) {
    if (prompt.empty()) {
      send_error(res, 400, "the prompt is empty", "invalid_request_error");
      return false;
    }
    if (static_cast<int>(prompt.size()) >= engine.max_seq_len()) {
      send_error(res, 400,
                 "this model's context is " + std::to_string(engine.max_seq_len()) +
                     " tokens, but the prompt has " + std::to_string(prompt.size()),
                 "invalid_request_error", "context_length_exceeded");
      return false;
    }
    return true;
  }

  // Non-streaming: wait for the whole completion.
  static std::string collect(Channel& ch) {
    std::string text;
    std::vector<std::string> pieces;
    while (ch.next(pieces)) {
      for (auto& p : pieces) text += p;
    }
    return text;
  }

  // Streaming: one SSE event per text piece. `chunk(text, finish)` builds an event payload;
  // `first` (may be null) is sent before any text, `usage_object` names the usage chunk.
  void stream(httplib::Response& res, std::shared_ptr<Channel> ch, openai::ResponseMeta meta,
              bool include_usage, bool chat) {
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [ch, meta, include_usage, chat](size_t, httplib::DataSink& sink) {
          auto send = [&](const std::string& s) {
            if (sink.write(s.data(), s.size())) return true;
            ch->cancelled = true;  // the client went away: stop generating
            return false;
          };
          auto event = [&](const std::string& text, std::optional<FinishReason> finish) {
            return send(sse::event(chat ? openai::chat_chunk(meta, json{{"content", text}}, finish)
                                        : openai::completion_chunk(meta, text, finish)));
          };
          if (chat && !send(sse::event(openai::chat_chunk(
                          meta, json{{"role", "assistant"}, {"content", ""}}, std::nullopt)))) {
            return false;
          }
          std::vector<std::string> pieces;
          while (ch->next(pieces)) {
            for (const auto& p : pieces) {
              if (!event(p, std::nullopt)) return false;
            }
          }
          if (!ch->error.empty()) {
            send(sse::event(openai::error_body(ch->error, "server_error")));
          } else {
            const auto finish = ch->result.finish_reason;
            if (chat) {
              if (!send(sse::event(openai::chat_chunk(meta, json::object(), finish)))) return false;
            } else if (!event("", finish)) {
              return false;
            }
            if (include_usage &&
                !send(sse::event(openai::usage_chunk(
                    meta, chat ? "chat.completion.chunk" : "text_completion", ch->result)))) {
              return false;
            }
          }
          send(sse::done());
          sink.done();
          return true;
        },
        [ch](bool success) {
          if (!success) ch->cancelled = true;
        });
  }

  void handle_chat(const httplib::Request& req, httplib::Response& res) {
    const auto parsed = openai::parse_chat_request(json::parse(req.body));
    if (engine.chat_template().is_raw()) {
      send_error(res, 400,
                 "model '" + engine.model_name() +
                     "' has no chat template; use /v1/completions instead",
                 "invalid_request_error");
      return;
    }
    const std::string text = engine.chat_template().render(parsed.messages);
    std::vector<int32_t> prompt = engine.tokenizer().encode(text);
    if (!check_length(prompt, res)) return;

    const auto meta = openai::new_meta("chatcmpl", engine.model_name());
    auto ch = start(std::move(prompt), parsed.sampling, "chat");
    if (parsed.stream) {
      stream(res, ch, meta, parsed.include_usage, /*chat=*/true);
      return;
    }
    const std::string content = collect(*ch);
    if (!ch->error.empty()) return send_error(res, 500, ch->error, "server_error");
    send_json(res, 200, openai::chat_response(meta, content, ch->result));
  }

  void handle_completion(const httplib::Request& req, httplib::Response& res) {
    const auto parsed = openai::parse_completion_request(json::parse(req.body));
    std::vector<int32_t> prompt;
    // Match HF's default add_special_tokens=True: BOS for models that want it (Llama).
    const Tokenizer& tok = engine.tokenizer();
    if (tok.add_bos() && tok.bos_id() >= 0) prompt.push_back(tok.bos_id());
    const auto ids = tok.encode(parsed.prompt);
    prompt.insert(prompt.end(), ids.begin(), ids.end());
    if (!check_length(prompt, res)) return;

    const auto meta = openai::new_meta("cmpl", engine.model_name());
    auto ch = start(std::move(prompt), parsed.sampling, "completion");
    if (parsed.stream) {
      stream(res, ch, meta, parsed.include_usage, /*chat=*/false);
      return;
    }
    const std::string text = collect(*ch);
    if (!ch->error.empty()) return send_error(res, 500, ch->error, "server_error");
    send_json(res, 200, openai::completion_response(meta, text, ch->result));
  }

  void register_routes() {
    using httplib::Request;
    using httplib::Response;

    http.set_pre_routing_handler([this](const Request& req, Response& res) {
      if (config.api_key.empty() || req.path.rfind("/v1/", 0) != 0) {
        return httplib::Server::HandlerResponse::Unhandled;
      }
      if (req.get_header_value("Authorization") != "Bearer " + config.api_key) {
        send_error(res, 401, "invalid or missing API key", "invalid_request_error",
                   "invalid_api_key");
        return httplib::Server::HandlerResponse::Handled;
      }
      return httplib::Server::HandlerResponse::Unhandled;
    });

    // Bad input surfaces as exceptions from parsing: map them to OpenAI-style errors.
    auto guarded = [](auto handler) {
      return [handler](const Request& req, Response& res) {
        try {
          handler(req, res);
        } catch (const json::exception& e) {
          send_error(res, 400, std::string("invalid JSON body: ") + e.what(),
                     "invalid_request_error");
        } catch (const std::invalid_argument& e) {
          send_error(res, 400, e.what(), "invalid_request_error");
        }
      };
    };
    http.Post("/v1/chat/completions",
              guarded([this](const Request& q, Response& r) { handle_chat(q, r); }));
    http.Post("/v1/completions",
              guarded([this](const Request& q, Response& r) { handle_completion(q, r); }));
    http.Get("/v1/models", [this](const Request&, Response& res) {
      const auto meta = openai::new_meta("model", engine.model_name());
      send_json(res, 200,
                {{"object", "list"},
                 {"data", json::array({{{"id", engine.model_name()},
                                        {"object", "model"},
                                        {"created", meta.created},
                                        {"owned_by", "ography"},
                                        {"max_model_len", engine.max_seq_len()}}})}});
    });
    http.Get("/health", [this](const Request&, Response& res) {
      send_json(res, 200,
                {{"status", "ok"},
                 {"model", engine.model_name()},
                 {"requests", requests.load()},
                 {"in_flight", in_flight.load()},
                 {"cancelled", cancelled.load()}});
    });

    http.set_exception_handler([](const Request&, Response& res, std::exception_ptr ep) {
      std::string what = "internal error";
      try {
        std::rethrow_exception(ep);
      } catch (const std::exception& e) {
        what = e.what();
      } catch (...) {
      }
      send_error(res, 500, what, "server_error");
    });
    http.set_error_handler([](const Request& req, Response& res) {
      if (res.body.empty()) {
        send_error(res, res.status, "no route for " + req.method + " " + req.path,
                   "invalid_request_error");
      }
    });
  }
};

HttpServer::HttpServer(Engine& engine, ServerConfig config)
    : impl_(std::make_unique<Impl>(engine, std::move(config))) {
  impl_->register_routes();
}

HttpServer::~HttpServer() {
  stop();
  // Impl's RequestQueue finishes any queued generations before the engine is released.
}

int HttpServer::bind() {
  Impl& m = *impl_;
  if (m.bound) return m.port;
  if (m.config.port == 0) {
    m.port = m.http.bind_to_any_port(m.config.host);
    if (m.port < 0) throw std::runtime_error("cannot bind " + m.config.host);
  } else {
    if (!m.http.bind_to_port(m.config.host, m.config.port)) {
      throw std::runtime_error("cannot bind " + m.config.host + ":" +
                               std::to_string(m.config.port) + " (in use?)");
    }
    m.port = m.config.port;
  }
  m.bound = true;
  return m.port;
}

void HttpServer::run() {
  bind();
  impl_->http.listen_after_bind();
}

void HttpServer::stop() { impl_->http.stop(); }

}  // namespace ography
