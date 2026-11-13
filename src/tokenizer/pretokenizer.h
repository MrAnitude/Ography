#pragma once

#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace ography {

// The pretokenizer split regexes used by byte-level BPE models, implemented by hand
// (std::regex is far too slow and has no \p{L}/\p{N}).
enum class PretokenizerKind {
  // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
  Gpt2,
  // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*
  //   |\s*[\r\n]+|\s+(?!\S)|\s+
  Llama3,
  // As Llama3, but \p{N} (one digit per piece) instead of \p{N}{1,3}.
  Qwen2,
};

// Identifies the pretokenizer from tokenizer.json's "pre_tokenizer" section. Throws for
// configurations not implemented here.
PretokenizerKind detect_pretokenizer(const nlohmann::json& pre_tokenizer);

// Splits text into pieces (views into text), each of which is BPE-encoded independently.
// The pieces concatenate back to text exactly.
void pretokenize(PretokenizerKind kind, std::string_view text, std::vector<std::string_view>& out);

}  // namespace ography
