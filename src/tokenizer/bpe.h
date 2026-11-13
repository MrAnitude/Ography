#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace ography {

// Byte-level BPE core: vocab, merge ranks, byte<->unicode mapping, pretokenizer.
// Internal to the tokenizer module.
class BpeModel {
 public:
  explicit BpeModel(const nlohmann::json& tokenizer_json);

  // Encodes text with no special tokens in it.
  void encode_ordinary(std::string_view text, std::vector<int32_t>& out) const;
  // Raw bytes for a token (after reversing the byte-level unicode mapping).
  const std::string& token_bytes(int32_t id) const;

  std::unordered_map<std::string, int32_t> special_tokens;

 private:
  // TODO: hand-written pretokenizer matching the model's split regex
  //       (std::regex is far too slow and lacks \p{L}/\p{N}).
  std::unordered_map<std::string, int32_t> vocab_;
  std::unordered_map<std::string, int32_t> merge_ranks_;  // key: "left right"
  std::vector<std::string> id_to_bytes_;
};

}  // namespace ography
