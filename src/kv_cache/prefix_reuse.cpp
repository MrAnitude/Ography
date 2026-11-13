#include "kv_cache/prefix_reuse.h"

#include "core/logging.h"

namespace ography {

size_t PrefixTracker::reusable_prefix(const std::vector<int32_t>& prompt) const {
  // TODO: longest common prefix; always leave >= 1 token to prefill so we get fresh logits.
  OGRAPHY_TODO("PrefixTracker::reusable_prefix");
}

void PrefixTracker::set_cached(std::vector<int32_t> tokens) { cached_ = std::move(tokens); }

}  // namespace ography
