#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ography {

// Per-request generation settings (mirrors the OpenAI request fields).
struct SamplingParams {
  float temperature = 1.0f;  // 0 => greedy
  int top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float repetition_penalty = 1.0f;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  uint64_t seed = 0;
  int max_tokens = 512;
  std::vector<std::string> stop;      // stop strings
  std::vector<int32_t> stop_token_ids;  // in addition to the model's EOS ids
  bool ignore_eos = false;  // keep going past the model's EOS ids (benchmarks)
};

}  // namespace ography
