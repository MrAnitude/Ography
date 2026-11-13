#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace ography::unicode {

// Regex character classes used by the pretokenizers (Unicode 15 general categories).
bool is_letter(char32_t cp);      // \p{L}
bool is_number(char32_t cp);      // \p{N}
bool is_whitespace(char32_t cp);  // \s (White_Space property)

// Decodes the code point starting at s[i] and returns its length in bytes (1..4). Malformed
// input (bad lead byte, truncated or overlong sequence, surrogate) decodes as U+FFFD with
// length 1, so every byte is consumed exactly once.
int decode_utf8(std::string_view s, size_t i, char32_t& cp);

void append_utf8(char32_t cp, std::string& out);

// Length of the longest prefix of s that is valid UTF-8 up to a possibly incomplete final
// character. If s ends partway through a well-formed multi-byte sequence, that tail is
// excluded; malformed bytes are included (the caller replaces them).
size_t complete_utf8_prefix(std::string_view s);

// s with every malformed byte replaced by U+FFFD.
std::string sanitize_utf8(std::string_view s);

}  // namespace ography::unicode
