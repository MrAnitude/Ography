#include "tokenizer/pretokenizer.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "tokenizer/unicode.h"

namespace ography {

namespace {

const char* const kGpt2Regex =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
const char* const kLlama3Regex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
const char* const kQwen2Regex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

enum Class : uint8_t { kLetter, kNumber, kSpace, kOther };

// The text decoded once into code points, with their classes and byte offsets.
struct Chars {
  std::vector<char32_t> cp;
  std::vector<uint8_t> cls;
  std::vector<size_t> offset;  // offset[i] = byte offset of cp[i]; offset[n] = text size

  explicit Chars(std::string_view text) {
    for (size_t i = 0; i < text.size();) {
      char32_t c;
      const int len = unicode::decode_utf8(text, i, c);
      cp.push_back(c);
      cls.push_back(unicode::is_letter(c)   ? kLetter
                    : unicode::is_number(c) ? kNumber
                    : unicode::is_whitespace(c) ? kSpace
                                                : kOther);
      offset.push_back(i);
      i += len;
    }
    offset.push_back(text.size());
  }
  size_t size() const { return cp.size(); }
};

// Number of code points from i while cls == c.
size_t run(const Chars& s, size_t i, Class c) {
  size_t j = i;
  while (j < s.size() && s.cls[j] == c) ++j;
  return j - i;
}

char32_t lower_ascii(char32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// 's|'t|'re|'ve|'m|'ll|'d, optionally case-insensitive. Returns code points matched.
size_t contraction(const Chars& s, size_t i, bool ignore_case) {
  if (s.cp[i] != '\'' || i + 1 >= s.size()) return 0;
  auto at = [&](size_t k) -> char32_t {
    if (i + k >= s.size()) return 0;
    char32_t c = s.cp[i + k];
    if (ignore_case) {
      c = lower_ascii(c);
      if (c == 0x17F) c = 's';  // LATIN SMALL LETTER LONG S case-folds to 's'
    }
    return c;
  };
  const char32_t a = at(1);
  if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
  const char32_t b = at(2);
  if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
  return 0;
}

// \s+(?!\S)|\s+ at i (s.cls[i] == kSpace): a whitespace run, leaving its last character for
// the next piece when a non-space follows and the run is longer than one.
size_t trailing_space_run(const Chars& s, size_t i) {
  const size_t len = run(s, i, kSpace);
  if (i + len == s.size() || len == 1) return len;
  return len - 1;
}

size_t match_gpt2(const Chars& s, size_t i) {
  if (size_t n = contraction(s, i, false)) return n;
  // ' ?\p{L}+', ' ?\p{N}+', ' ?[^\s\p{L}\p{N}]+'
  const size_t j = (s.cp[i] == ' ' && i + 1 < s.size()) ? i + 1 : i;
  for (Class c : {kLetter, kNumber, kOther}) {
    if (s.cls[j] == c) return (j - i) + run(s, j, c);
  }
  return trailing_space_run(s, i);  // only whitespace is left
}

bool is_newline(char32_t c) { return c == '\r' || c == '\n'; }

size_t match_llama3(const Chars& s, size_t i, size_t max_digits) {
  if (size_t n = contraction(s, i, true)) return n;
  // [^\r\n\p{L}\p{N}]?\p{L}+
  if (s.cls[i] == kLetter) return run(s, i, kLetter);
  if (s.cls[i] != kNumber && !is_newline(s.cp[i]) && i + 1 < s.size() && s.cls[i + 1] == kLetter) {
    return 1 + run(s, i + 1, kLetter);
  }
  // \p{N}{1,max_digits}
  if (s.cls[i] == kNumber) return std::min(run(s, i, kNumber), max_digits);
  // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
  const size_t j = (s.cp[i] == ' ' && i + 1 < s.size()) ? i + 1 : i;
  if (s.cls[j] == kOther) {
    size_t k = j + run(s, j, kOther);
    while (k < s.size() && is_newline(s.cp[k])) ++k;
    return k - i;
  }
  // \s*[\r\n]+ : the whitespace run up to and including its last newline.
  const size_t len = run(s, i, kSpace);
  for (size_t k = i + len; k > i; --k) {
    if (is_newline(s.cp[k - 1])) return k - i;
  }
  return trailing_space_run(s, i);
}

}  // namespace

PretokenizerKind detect_pretokenizer(const nlohmann::json& pre) {
  const std::string type = pre.value("type", "");
  if (type == "ByteLevel") {
    if (!pre.value("use_regex", true) || pre.value("add_prefix_space", false)) {
      throw std::runtime_error("tokenizer: unsupported ByteLevel pre_tokenizer options");
    }
    return PretokenizerKind::Gpt2;
  }
  if (type == "Sequence") {
    const auto& seq = pre.at("pretokenizers");
    if (seq.size() == 2 && seq[0].value("type", "") == "Split" &&
        seq[0].value("behavior", "") == "Isolated" && !seq[0].value("invert", false) &&
        seq[1].value("type", "") == "ByteLevel" && !seq[1].value("use_regex", true) &&
        !seq[1].value("add_prefix_space", false)) {
      const std::string regex = seq[0].at("pattern").value("Regex", "");
      if (regex == kLlama3Regex) return PretokenizerKind::Llama3;
      if (regex == kQwen2Regex) return PretokenizerKind::Qwen2;
      throw std::runtime_error("tokenizer: unsupported split regex: " + regex);
    }
  }
  throw std::runtime_error("tokenizer: unsupported pre_tokenizer: " + pre.dump());
}

void pretokenize(PretokenizerKind kind, std::string_view text, std::vector<std::string_view>& out) {
  const Chars s(text);
  for (size_t i = 0; i < s.size();) {
    size_t n = 0;
    switch (kind) {
      case PretokenizerKind::Gpt2: n = match_gpt2(s, i); break;
      case PretokenizerKind::Llama3: n = match_llama3(s, i, 3); break;
      case PretokenizerKind::Qwen2: n = match_llama3(s, i, 1); break;
    }
    out.push_back(text.substr(s.offset[i], s.offset[i + n] - s.offset[i]));
    i += n;
  }
}

}  // namespace ography
