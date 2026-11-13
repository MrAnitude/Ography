#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ography {

class BpeModel;

// Loads a HF tokenizer.json (byte-level BPE: Llama 3, Qwen, Mistral v3+).
// TODO: SentencePiece-style BPE (Llama 2 / Mistral v1) via the same file's "model" section.
class Tokenizer {
 public:
  static Tokenizer from_file(const std::string& tokenizer_json_path);
  ~Tokenizer();
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;

  // Special tokens like "<|eot_id|>" written in the text are matched as single tokens.
  std::vector<int32_t> encode(std::string_view text) const;
  std::string decode(const std::vector<int32_t>& ids) const;

  int32_t token_id(std::string_view token) const;  // -1 if absent
  int32_t vocab_size() const;
  int32_t bos_id() const;
  std::vector<int32_t> eos_ids() const;  // models often have several (e.g. eot + eos)

 private:
  Tokenizer();
  std::unique_ptr<BpeModel> bpe_;
};

// Turns a stream of token ids into text pieces that are always valid UTF-8.
// A multi-byte character can span several tokens, so bytes are held back until complete.
class IncrementalDecoder {
 public:
  explicit IncrementalDecoder(const Tokenizer& tokenizer);
  std::string push(int32_t token_id);  // returns newly completed text (may be empty)
  std::string flush();

 private:
  const Tokenizer& tokenizer_;
  std::string pending_bytes_;
};

}  // namespace ography
