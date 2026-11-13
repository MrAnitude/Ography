// End-to-end latency benchmark.
//
//   ography_bench --model /path/to/hf-model [--prompt-len 512] [--gen-len 256] [--iters 5]
//                 [--fp8-kv] [--no-cuda-graphs] [--max-seq-len N]
//
// Feeds a random prompt of --prompt-len tokens (cold cache each iteration) and generates
// --gen-len tokens greedily, ignoring EOS. Reports the median time to first token (prefill),
// decode tokens/s and ms/token, and the GPU memory in use, so every optimisation can be
// measured against a baseline.

#include <algorithm>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "engine/engine.h"
#include "tokenizer/tokenizer.h"

int main(int argc, char** argv) {
  using namespace ography;
  EngineConfig cfg;
  int prompt_len = 512, gen_len = 256, iters = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") cfg.model_dir = next();
    else if (a == "--prompt-len") prompt_len = std::stoi(next());
    else if (a == "--gen-len") gen_len = std::stoi(next());
    else if (a == "--iters") iters = std::stoi(next());
    else if (a == "--fp8-kv") cfg.fp8_kv_cache = true;
    else if (a == "--no-cuda-graphs") cfg.use_cuda_graphs = false;
    else if (a == "--max-seq-len") cfg.max_seq_len = std::stoi(next());
    else {
      std::fprintf(stderr,
                   "usage: %s --model DIR [--prompt-len N] [--gen-len N] [--iters N] [--fp8-kv] "
                   "[--no-cuda-graphs] [--max-seq-len N]\n",
                   argv[0]);
      return 2;
    }
  }
  if (cfg.model_dir.empty() || prompt_len < 1 || gen_len < 1 || iters < 1) {
    std::fprintf(stderr, "--model is required; lengths and iters must be positive\n");
    return 2;
  }

  try {
    if (cfg.max_seq_len == 0) cfg.max_seq_len = prompt_len + gen_len + 64;
    Engine engine(cfg);
    // Ordinary tokens only: ids from the lower half of the vocabulary.
    std::mt19937 rng(42);
    std::vector<int32_t> prompt(prompt_len);
    for (auto& id : prompt) id = static_cast<int32_t>(rng() % (engine.tokenizer().vocab_size() / 2));

    SamplingParams sp;
    sp.temperature = 0;
    sp.max_tokens = gen_len;
    sp.ignore_eos = true;
    std::vector<double> ttft, tps;
    for (int it = 0; it < iters; ++it) {
      engine.clear_cache();
      const GenerationResult r = engine.generate({prompt, sp}, [](const TokenEvent&) { return true; });
      ttft.push_back(r.time_to_first_token_ms);
      tps.push_back(r.tokens_per_second);
    }
    std::sort(ttft.begin(), ttft.end());
    std::sort(tps.begin(), tps.end());
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const double med_tps = tps[tps.size() / 2];
    std::printf("%s: prompt %d, generate %d, median of %d\n", engine.model_name().c_str(),
                prompt_len, gen_len, iters);
    std::printf("  time to first token  %8.1f ms   (prefill %.0f tok/s)\n", ttft[ttft.size() / 2],
                prompt_len / (ttft[ttft.size() / 2] / 1e3));
    std::printf("  decode               %8.1f tok/s  (%.2f ms/token)\n", med_tps, 1e3 / med_tps);
    std::printf("  GPU memory in use    %8.0f MiB of %.0f\n", (total_b - free_b) / 1048576.0,
                total_b / 1048576.0);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
