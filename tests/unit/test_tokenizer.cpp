#include <gtest/gtest.h>

#include "tokenizer/tokenizer.h"

// TODO: point at a real tokenizer.json (env OGRAPHY_TOKENIZER) and compare encode() output
// with ids produced by HF `tokenizers` for a fixed set of strings (unicode, whitespace,
// numbers, code, special tokens). Also: decode(encode(s)) == s.
TEST(Tokenizer, MatchesHuggingFace) { GTEST_SKIP() << "tokenizer not implemented yet"; }

TEST(IncrementalDecoder, HoldsBackPartialUtf8) { GTEST_SKIP() << "not implemented yet"; }
