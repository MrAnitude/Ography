#include "tokenizer/tokenizer.h"

#include "core/logging.h"
#include "tokenizer/bpe.h"

namespace ography {

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer Tokenizer::from_file(const std::string& tokenizer_json_path) {
  OGRAPHY_TODO("Tokenizer::from_file");
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
  OGRAPHY_TODO("Tokenizer::encode (split on special tokens, pretokenize, BPE)");
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  OGRAPHY_TODO("Tokenizer::decode");
}

int32_t Tokenizer::token_id(std::string_view token) const { OGRAPHY_TODO("Tokenizer::token_id"); }
int32_t Tokenizer::vocab_size() const { OGRAPHY_TODO("Tokenizer::vocab_size"); }
int32_t Tokenizer::bos_id() const { OGRAPHY_TODO("Tokenizer::bos_id"); }
std::vector<int32_t> Tokenizer::eos_ids() const { OGRAPHY_TODO("Tokenizer::eos_ids"); }

IncrementalDecoder::IncrementalDecoder(const Tokenizer& tokenizer) : tokenizer_(tokenizer) {}

std::string IncrementalDecoder::push(int32_t token_id) {
  OGRAPHY_TODO("IncrementalDecoder::push");
}

std::string IncrementalDecoder::flush() { OGRAPHY_TODO("IncrementalDecoder::flush"); }

}  // namespace ography
