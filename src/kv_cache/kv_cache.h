#pragma once

#include <vector>

#include "core/memory.h"
#include "core/tensor.h"

namespace ography {

struct KVCacheConfig {
  int num_layers = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  int max_seq_len = 0;          // sized at startup from free VRAM after weights load
  DType dtype = DType::F16;     // or FP8_E4M3 (unscaled): half the memory, twice the context
  int num_q_heads = 0;          // query heads; if set, decode-attention scratch is allocated
};

// One contiguous [max_seq_len, num_kv_heads, head_dim] K and V buffer per layer.
// Single client means single sequence, so no paging/block tables are needed.
class KVCache {
 public:
  KVCache(const KVCacheConfig& config, DeviceArena& arena);

  Tensor& k(int layer) { return k_[layer]; }
  Tensor& v(int layer) { return v_[layer]; }

  int seq_len() const { return seq_len_; }   // number of valid cached positions
  int capacity() const { return config_.max_seq_len; }
  void advance(int n_tokens);                // after a forward pass writes n_tokens
  void truncate(int n_tokens);               // keep a prefix, drop the rest (prefix reuse)
  void clear() { truncate(0); }

  static size_t bytes_per_token(const KVCacheConfig& config);

  // Scratch for kernels::attention_decode over this cache (empty if num_q_heads wasn't set,
  // in which case decode attention falls back to the naive kernel).
  Tensor& decode_scratch() { return decode_scratch_; }
  // Bytes of decode scratch a cache of this config allocates.
  static size_t scratch_bytes(const KVCacheConfig& config);

 private:
  KVCacheConfig config_;
  std::vector<Tensor> k_, v_;
  Tensor decode_scratch_;
  int seq_len_ = 0;
};

}  // namespace ography
