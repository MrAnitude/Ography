#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json_fwd.hpp>

#include "tokenizer/pretokenizer.h"

namespace ography {

// Byte-level BPE core: vocab, merge ranks, byte<->unicode mapping, pretokenizer, added
// tokens. Internal to the tokenizer module.
class BpeModel {
 public:
  explicit BpeModel(const nlohmann::json& tokenizer_json);

  // Encodes text, matching added tokens (e.g. "<|eot_id|>") written in it as single tokens.
  void encode(std::string_view text, std::vector<int32_t>& out) const;
  // Encodes text with no added-token matching.
  void encode_ordinary(std::string_view text, std::vector<int32_t>& out) const;

  // Raw bytes for a token (after reversing the byte-level unicode mapping).
  const std::string& token_bytes(int32_t id) const { return id_to_bytes_.at(id); }
  // Id of a token by its tokenizer.json spelling (e.g. "Ġthe" or "<|eot_id|>"); -1 if absent.
  int32_t token_id(std::string_view token) const;
  int32_t vocab_size() const { return static_cast<int32_t>(id_to_bytes_.size()); }

 private:
  struct Merge {
    int32_t rank;
    int32_t merged_id;
  };
  static uint64_t pair_key(int32_t a, int32_t b) {
    return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
  }

  void bpe(std::string_view piece, std::vector<int32_t>& out) const;

  PretokenizerKind pretokenizer_;
  bool ignore_merges_ = false;
  std::unordered_map<std::string, int32_t> vocab_;  // byte-level-mapped spelling -> id
  std::unordered_map<std::string, int32_t> added_;  // literal content -> id
  // Added tokens bucketed by first byte, longest first, for leftmost-longest matching.
  std::vector<std::vector<std::pair<std::string, int32_t>>> added_by_first_byte_;
  std::unordered_map<uint64_t, Merge> merges_;
  int32_t byte_ids_[256];          // id of the single-byte token for each byte
  std::string byte_to_unicode_[256];  // GPT-2's printable stand-in for each byte
  std::vector<std::string> id_to_bytes_;
};

}  // namespace ography
