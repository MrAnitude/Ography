#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "engine/sampling_params.h"

namespace ography {

enum class FinishReason { Stop, Length, Cancelled };

// Detects stop tokens, stop strings and the token limit. Stop strings can span token
// boundaries, so text that might be the start of a stop string is held back from the stream
// until it is either confirmed or ruled out. Neither stop strings nor stop tokens' own text
// are ever emitted (OpenAI semantics).
class StopChecker {
 public:
  StopChecker(const SamplingParams& params, std::vector<int32_t> eos_ids);

  struct Result {
    std::string emit;                    // text safe to stream to the client now
    std::optional<FinishReason> finish;  // set once generation should end
  };

  // Called once per generated token with that token's decoded text (empty for stop tokens,
  // whose text is never shown). tokens_generated counts this token.
  Result push(int32_t token_id, const std::string& text_piece, int tokens_generated);

  // EOS ids or request stop_token_ids.
  bool is_stop_token(int32_t token_id) const;

 private:
  SamplingParams params_;
  std::vector<int32_t> stop_ids_;
  std::string held_back_;
};

}  // namespace ography
