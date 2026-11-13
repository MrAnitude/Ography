#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "tokenizer/bpe.h"
#include "tokenizer/unicode.h"

namespace ography {

namespace fs = std::filesystem;

namespace {

nlohmann::json read_json(const fs::path& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path.string());
  return nlohmann::json::parse(f);
}

// tokenizer_config.json spells tokens as "<s>" or {"content": "<s>", ...}.
std::string token_text(const nlohmann::json& j) {
  if (j.is_string()) return j.get<std::string>();
  if (j.is_object()) return j.value("content", "");
  return "";
}

}  // namespace

Tokenizer::Tokenizer() = default;
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer Tokenizer::from_file(const std::string& tokenizer_json_path) {
  Tokenizer t;
  t.bpe_ = std::make_unique<BpeModel>(read_json(tokenizer_json_path));

  // generation_config.json is what generate() uses, so it wins; tokenizer_config.json fills in.
  const fs::path dir = fs::path(tokenizer_json_path).parent_path();
  auto add_eos = [&](int32_t id) {
    if (id >= 0 && !t.is_eos(id)) t.eos_ids_.push_back(id);
  };
  if (fs::exists(dir / "generation_config.json")) {
    const auto g = read_json(dir / "generation_config.json");
    if (g.contains("bos_token_id") && g["bos_token_id"].is_number_integer()) {
      t.bos_id_ = g["bos_token_id"].get<int32_t>();
    }
    if (g.contains("eos_token_id")) {
      const auto& e = g["eos_token_id"];
      if (e.is_number_integer()) add_eos(e.get<int32_t>());
      if (e.is_array()) for (const auto& id : e) add_eos(id.get<int32_t>());
    }
  }
  if (fs::exists(dir / "tokenizer_config.json")) {
    const auto c = read_json(dir / "tokenizer_config.json");
    if (t.bos_id_ < 0 && c.contains("bos_token")) t.bos_id_ = t.token_id(token_text(c["bos_token"]));
    if (c.contains("eos_token")) add_eos(t.token_id(token_text(c["eos_token"])));
    if (c.contains("add_bos_token") && c["add_bos_token"].is_boolean()) {
      t.add_bos_ = c["add_bos_token"].get<bool>();
    }
  }
  return t;
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
  std::vector<int32_t> ids;
  bpe_->encode(text, ids);
  return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string out;
  for (int32_t id : ids) out += bpe_->token_bytes(id);
  return out;
}

const std::string& Tokenizer::token_bytes(int32_t id) const { return bpe_->token_bytes(id); }
int32_t Tokenizer::token_id(std::string_view token) const { return bpe_->token_id(token); }
int32_t Tokenizer::vocab_size() const { return bpe_->vocab_size(); }

bool Tokenizer::is_eos(int32_t id) const {
  return std::find(eos_ids_.begin(), eos_ids_.end(), id) != eos_ids_.end();
}

IncrementalDecoder::IncrementalDecoder(const Tokenizer& tokenizer) : tokenizer_(tokenizer) {}

std::string IncrementalDecoder::push(int32_t token_id) {
  pending_bytes_ += tokenizer_.token_bytes(token_id);
  const size_t n = unicode::complete_utf8_prefix(pending_bytes_);
  std::string out = unicode::sanitize_utf8(std::string_view(pending_bytes_).substr(0, n));
  pending_bytes_.erase(0, n);
  return out;
}

std::string IncrementalDecoder::flush() {
  std::string out = unicode::sanitize_utf8(pending_bytes_);
  pending_bytes_.clear();
  return out;
}

}  // namespace ography
