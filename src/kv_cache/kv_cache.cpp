#include "kv_cache/kv_cache.h"

#include <stdexcept>
#include <string>

namespace ography {

KVCache::KVCache(const KVCacheConfig& config, DeviceArena& arena) : config_(config) {
  if (config.num_layers <= 0 || config.num_kv_heads <= 0 || config.head_dim <= 0 ||
      config.max_seq_len <= 0) {
    throw std::invalid_argument("KVCache: all dimensions must be positive");
  }
  if (config.dtype != DType::F16) {
    throw std::invalid_argument("KVCache: only f16 is supported so far (fp8 is roadmap step 9)");
  }
  k_.reserve(config.num_layers);
  v_.reserve(config.num_layers);
  const std::vector<int64_t> shape = {config.max_seq_len, config.num_kv_heads, config.head_dim};
  for (int i = 0; i < config.num_layers; ++i) {
    k_.push_back(arena.allocate(config.dtype, shape));
    v_.push_back(arena.allocate(config.dtype, shape));
  }
}

void KVCache::advance(int n_tokens) {
  if (n_tokens < 0 || seq_len_ + n_tokens > config_.max_seq_len) {
    throw std::out_of_range("KVCache::advance by " + std::to_string(n_tokens) + " from " +
                            std::to_string(seq_len_) + " exceeds capacity " +
                            std::to_string(config_.max_seq_len));
  }
  seq_len_ += n_tokens;
}

void KVCache::truncate(int n_tokens) {
  if (n_tokens < 0 || n_tokens > seq_len_) {
    throw std::out_of_range("KVCache::truncate to " + std::to_string(n_tokens) +
                            " but only " + std::to_string(seq_len_) + " are cached");
  }
  seq_len_ = n_tokens;
}

size_t KVCache::bytes_per_token(const KVCacheConfig& config) {
  return 2 * static_cast<size_t>(config.num_layers) * config.num_kv_heads * config.head_dim *
         dtype_bits(config.dtype) / 8;
}

}  // namespace ography
