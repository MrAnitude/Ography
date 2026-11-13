#include "tokenizer/unicode.h"

#include <algorithm>
#include <iterator>

namespace ography::unicode {

namespace {

struct CodepointRange {
  char32_t first;
  char32_t last;
};

#include "tokenizer/unicode_tables.inc"

template <size_t N>
bool in_ranges(const CodepointRange (&table)[N], char32_t cp) {
  const auto* it = std::upper_bound(std::begin(table), std::end(table), cp,
                                    [](char32_t c, const CodepointRange& r) { return c < r.first; });
  return it != std::begin(table) && cp <= std::prev(it)->last;
}

constexpr char32_t kReplacement = 0xFFFD;

// Expected sequence length for a lead byte, and the valid range of the second byte
// (which excludes overlongs, surrogates and code points above U+10FFFF). 0 = invalid lead.
int sequence_length(unsigned char b, unsigned char& lo, unsigned char& hi) {
  lo = 0x80;
  hi = 0xBF;
  if (b < 0x80) return 1;
  if (b < 0xC2) return 0;  // continuation byte, or overlong C0/C1
  if (b < 0xE0) return 2;
  if (b < 0xF0) {
    if (b == 0xE0) lo = 0xA0;
    if (b == 0xED) hi = 0x9F;
    return 3;
  }
  if (b < 0xF5) {
    if (b == 0xF0) lo = 0x90;
    if (b == 0xF4) hi = 0x8F;
    return 4;
  }
  return 0;
}

}  // namespace

bool is_letter(char32_t cp) {
  if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
  return in_ranges(kLetterRanges, cp);
}

bool is_number(char32_t cp) {
  if (cp < 0x80) return cp >= '0' && cp <= '9';
  return in_ranges(kNumberRanges, cp);
}

bool is_whitespace(char32_t cp) {
  if (cp < 0x80) return cp == ' ' || (cp >= 0x09 && cp <= 0x0D);
  return in_ranges(kWhitespaceRanges, cp);
}

int decode_utf8(std::string_view s, size_t i, char32_t& cp) {
  const auto b0 = static_cast<unsigned char>(s[i]);
  unsigned char lo, hi;
  const int len = sequence_length(b0, lo, hi);
  if (len == 1) {
    cp = b0;
    return 1;
  }
  if (len == 0 || i + len > s.size()) {
    cp = kReplacement;
    return 1;
  }
  char32_t c = b0 & (0xFF >> (len + 1));
  for (int k = 1; k < len; ++k) {
    const auto b = static_cast<unsigned char>(s[i + k]);
    if (b < (k == 1 ? lo : 0x80) || b > (k == 1 ? hi : 0xBF)) {
      cp = kReplacement;
      return 1;
    }
    c = (c << 6) | (b & 0x3F);
  }
  cp = c;
  return len;
}

void append_utf8(char32_t cp, std::string& out) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

size_t complete_utf8_prefix(std::string_view s) {
  // Only the last few bytes can be an incomplete sequence; check from the last lead byte.
  const size_t n = s.size();
  for (size_t back = 1; back <= std::min<size_t>(3, n); ++back) {
    const size_t i = n - back;
    const auto b = static_cast<unsigned char>(s[i]);
    if ((b & 0xC0) == 0x80) continue;  // continuation byte: keep looking for the lead
    unsigned char lo, hi;
    const int len = sequence_length(b, lo, hi);
    if (len <= static_cast<int>(back)) return n;  // complete, ASCII, or malformed lead
    // Incomplete: hold it back only if the bytes present so far are a valid start.
    for (size_t k = 1; k < back; ++k) {
      const auto c = static_cast<unsigned char>(s[i + k]);
      if (c < (k == 1 ? lo : 0x80) || c > (k == 1 ? hi : 0xBF)) return n;
    }
    return i;
  }
  return n;
}

std::string sanitize_utf8(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    char32_t cp;
    const int len = decode_utf8(s, i, cp);
    if (cp == kReplacement && len == 1 && static_cast<unsigned char>(s[i]) >= 0x80) {
      append_utf8(kReplacement, out);
    } else {
      out.append(s.substr(i, len));
    }
    i += len;
  }
  return out;
}

}  // namespace ography::unicode
