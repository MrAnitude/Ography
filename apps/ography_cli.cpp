// Terminal chat against the engine directly, no HTTP. The quickest way to debug
// tokenizer / chat-template / sampling problems.
//
//   ography_cli --model /path/to/hf-model [--temperature 0.7] [--greedy] ...
//
// Models with a chat template get a multi-turn chat; models without one (GPT-2) get
// completion mode, where each line is a prompt to continue. Ctrl-C stops a reply; Ctrl-C at
// the prompt (or Ctrl-D) exits. Prints time-to-first-token and tokens/sec after each reply.

#include <signal.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "core/logging.h"
#include "engine/engine.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"

namespace {

using namespace ography;

std::atomic<bool> g_interrupted{false};

void on_sigint(int) { g_interrupted = true; }

struct Options {
  EngineConfig engine;
  SamplingParams sampling;
  std::string system;
  bool raw = false;
};

[[noreturn]] void usage(const char* argv0, int code) {
  std::fprintf(code ? stderr : stdout,
               "usage: %s --model DIR [options]\n"
               "  --temperature F         default 0.7 (0 = greedy)\n"
               "  --greedy                same as --temperature 0\n"
               "  --top-k N               default 0 (off)\n"
               "  --top-p F               default 0.95\n"
               "  --min-p F               default 0 (off)\n"
               "  --repetition-penalty F  default 1 (off)\n"
               "  --max-tokens N          per reply, default 512\n"
               "  --seed N                default random\n"
               "  --system TEXT           system prompt (chat mode)\n"
               "  --raw                   completion mode even if the model has a chat template\n"
               "  --max-seq-len N         context length, default as large as fits\n"
               "  --prefill-chunk N       default 2048\n"
               "  --no-cuda-graphs        decode without graph replay (debugging)\n"
               "In chat: /reset clears the conversation, /quit exits.\n",
               argv0);
  std::exit(code);
}

Options parse(int argc, char** argv) {
  Options o;
  o.sampling.temperature = 0.7f;
  o.sampling.top_p = 0.95f;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage(argv[0], 2);
      return argv[++i];
    };
    if (a == "--model") o.engine.model_dir = next();
    else if (a == "--temperature") o.sampling.temperature = std::stof(next());
    else if (a == "--greedy") o.sampling.temperature = 0;
    else if (a == "--top-k") o.sampling.top_k = std::stoi(next());
    else if (a == "--top-p") o.sampling.top_p = std::stof(next());
    else if (a == "--min-p") o.sampling.min_p = std::stof(next());
    else if (a == "--repetition-penalty") o.sampling.repetition_penalty = std::stof(next());
    else if (a == "--max-tokens") o.sampling.max_tokens = std::stoi(next());
    else if (a == "--seed") o.sampling.seed = std::stoull(next());
    else if (a == "--system") o.system = next();
    else if (a == "--raw") o.raw = true;
    else if (a == "--max-seq-len") o.engine.max_seq_len = std::stoi(next());
    else if (a == "--prefill-chunk") o.engine.prefill_chunk = std::stoi(next());
    else if (a == "--no-cuda-graphs") o.engine.use_cuda_graphs = false;
    else if (a == "-h" || a == "--help") usage(argv[0], 0);
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage(argv[0], 2);
    }
  }
  if (o.engine.model_dir.empty()) usage(argv[0], 2);
  return o;
}

const char* finish_name(FinishReason r) {
  switch (r) {
    case FinishReason::Stop: return "stop";
    case FinishReason::Length: return "length";
    case FinishReason::Cancelled: return "cancelled";
  }
  return "?";
}

// Streams one reply to stdout; returns its text.
std::string reply(Engine& engine, const std::vector<int32_t>& prompt, const SamplingParams& sp,
                  bool tty) {
  std::string text;
  g_interrupted = false;
  const GenerationResult r = engine.generate({prompt, sp}, [&](const TokenEvent& ev) {
    std::fwrite(ev.text.data(), 1, ev.text.size(), stdout);
    std::fflush(stdout);
    text += ev.text;
    return !g_interrupted.load();
  });
  std::printf("\n");
  std::fflush(stdout);
  std::fprintf(stderr, "%s[%d+%d tokens (%d cached), TTFT %.0f ms, %.1f tok/s, %s]%s\n",
               tty ? "\033[2m" : "", r.prompt_tokens, r.completion_tokens, r.cached_prompt_tokens,
               r.time_to_first_token_ms, r.tokens_per_second, finish_name(r.finish_reason),
               tty ? "\033[0m" : "");
  return text;
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse(argc, argv);

  // No SA_RESTART: Ctrl-C at the prompt interrupts the blocking read so we can exit.
  struct sigaction sa {};
  sa.sa_handler = on_sigint;
  sigaction(SIGINT, &sa, nullptr);

  try {
    Engine engine(opt.engine);
    const bool chat = !opt.raw && !engine.chat_template().is_raw();
    const bool tty = isatty(fileno(stdin)) && isatty(fileno(stderr));
    std::fprintf(stderr, "%s: %s mode, %d-token context. Ctrl-C stops a reply, Ctrl-D exits.\n",
                 engine.model_name().c_str(),
                 chat ? engine.chat_template().format_name() : "completion",
                 engine.max_seq_len());

    std::vector<ChatMessage> history;
    if (!opt.system.empty()) history.push_back({"system", opt.system});

    std::string line;
    while (true) {
      if (tty) std::fprintf(stderr, chat ? "\n> " : "\n>> ");
      g_interrupted = false;
      if (!std::getline(std::cin, line) || g_interrupted) break;
      if (line.empty()) continue;
      if (line == "/quit" || line == "/exit") break;
      if (line == "/reset") {
        history.resize(opt.system.empty() ? 0 : 1);
        std::fprintf(stderr, "(conversation cleared)\n");
        continue;
      }

      try {
        if (!chat) {
          // Completion: echo nothing, just continue the text.
          if (tty) std::printf("%s", line.c_str());
          reply(engine, engine.tokenizer().encode(line), opt.sampling, tty);
          continue;
        }
        history.push_back({"user", line});
        const std::string prompt = engine.chat_template().render(history);
        const std::string answer =
            reply(engine, engine.tokenizer().encode(prompt), opt.sampling, tty);
        history.push_back({"assistant", answer});
      } catch (const std::invalid_argument& e) {
        // Typically the conversation outgrew the context.
        if (chat) history.pop_back();
        std::fprintf(stderr, "error: %s (try /reset)\n", e.what());
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
