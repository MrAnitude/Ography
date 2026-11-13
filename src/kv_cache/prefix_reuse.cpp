#include "kv_cache/prefix_reuse.h"

#include <algorithm>

namespace ography {

size_t PrefixTracker::reusable_prefix(const std::vector<int32_t>& prompt) const {
  const size_t limit = std::min(cached_.size(), prompt.size());
  size_t n = 0;
  while (n < limit && cached_[n] == prompt[n]) ++n;
  // Always prefill at least one token: the last prompt token's logits are what we sample from.
  if (n == prompt.size() && n > 0) --n;
  return n;
}

void PrefixTracker::set_cached(std::vector<int32_t> tokens) { cached_ = std::move(tokens); }

}  // namespace ography
