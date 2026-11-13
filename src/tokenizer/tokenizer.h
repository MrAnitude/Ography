#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ography {

class BpeModel;

// Loads a HF tokenizer.json (byte-level BPE: GPT-2, Llama 3, Qwen2). BOS/EOS ids come from
// generation_config.json and tokenizer_config.json in the same directory, when present.
// TODO: SentencePiece-style BPE (Llama 2 / Mistral v1) via the same file's "model" section.
class Tokenizer {
 public:
  static Tokenizer from_file(const std::string& tokenizer_json_path);
  ~Tokenizer();
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;

  // Special tokens like "<|eot_id|>" written in the text are matched as single tokens.
  // No BOS is added: chat templates write it into the text themselves.
  std::vector<int32_t> encode(std::string_view text) const;
  // Concatenated token bytes; may end in (or contain) incomplete UTF-8.
  std::string decode(const std::vector<int32_t>& ids) const;
  const std::string& token_bytes(int32_t id) const;

  int32_t token_id(std::string_view token) const;  // -1 if absent
  int32_t vocab_size() const;
  int32_t bos_id() const { return bos_id_; }  // -1 if the model has none
  const std::vector<int32_t>& eos_ids() const { return eos_ids_; }  // often several
  bool is_eos(int32_t id) const;
  // tokenizer_config.json "add_bos_token": whether raw text prompts should start with BOS
  // (true for Llama, false for Qwen and GPT-2).
  bool add_bos() const { return add_bos_; }

 private:
  Tokenizer();
  std::unique_ptr<BpeModel> bpe_;
  int32_t bos_id_ = -1;
  std::vector<int32_t> eos_ids_;
  bool add_bos_ = false;
};

// Turns a stream of token ids into text pieces that are always valid UTF-8.
// A multi-byte character can span several tokens, so bytes are held back until complete.
// Malformed bytes (which a model can emit) come out as U+FFFD.
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
