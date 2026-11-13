#include "kv_cache/kv_cache.h"

#include "core/logging.h"

namespace ography {

KVCache::KVCache(const KVCacheConfig& config, DeviceArena& arena) : config_(config) {
  OGRAPHY_TODO("KVCache: allocate per-layer K/V tensors from the arena");
}

void KVCache::advance(int n_tokens) {
  OGRAPHY_TODO("KVCache::advance");
}

void KVCache::truncate(int n_tokens) {
  OGRAPHY_TODO("KVCache::truncate");
}

size_t KVCache::bytes_per_token(const KVCacheConfig& config) {
  OGRAPHY_TODO("KVCache::bytes_per_token (2 * layers * kv_heads * head_dim * dtype size)");
}

}  // namespace ography
