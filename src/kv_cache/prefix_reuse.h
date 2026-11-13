#pragma once

#include <cstdint>
#include <vector>

namespace ography {

// Remembers which token ids are currently in the KV cache. In a multi-turn chat the new
// prompt usually starts with the entire previous conversation, so only the new suffix
// needs prefill: this is the main time-to-first-token win for a single client.
class PrefixTracker {
 public:
  // Number of leading tokens of `prompt` already cached and reusable.
  size_t reusable_prefix(const std::vector<int32_t>& prompt) const;

  // Record the cache contents after a request (prompt + generated tokens).
  void set_cached(std::vector<int32_t> tokens);
  void clear() { cached_.clear(); }

 private:
  std::vector<int32_t> cached_;
};

}  // namespace ography
