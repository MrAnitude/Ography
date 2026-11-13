// Tokenizer tests. Encodings are checked against HF `tokenizers` using fixtures dumped by
// tools/dump_tokenizer_cases.py into tests/data/tokenizer/<model>.json. Each fixture needs
// models/<model>/tokenizer.json (not checked in) and skips itself if that is absent.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

using namespace ography;
namespace fs = std::filesystem;

namespace {

const std::string kSource = OGRAPHY_SOURCE_DIR;

std::string tokenizer_path(const std::string& model) {
  return kSource + "/models/" + model + "/tokenizer.json";
}

// Loads models/<model>/tokenizer.json, or skips the test if absent.
#define LOAD_TOKENIZER_OR_SKIP(var, model)                                       \
  if (!fs::exists(tokenizer_path(model))) {                                      \
    GTEST_SKIP() << tokenizer_path(model) << " not found";                       \
  }                                                                              \
  const Tokenizer var = Tokenizer::from_file(tokenizer_path(model))

std::string show_ids(const std::vector<int32_t>& ids) {
  std::string s;
  for (int32_t id : ids) s += std::to_string(id) + " ";
  return s;
}

bool valid_utf8(const std::string& s) { return unicode::sanitize_utf8(s) == s; }

class MatchesHuggingFace : public ::testing::TestWithParam<std::string> {};

}  // namespace

TEST_P(MatchesHuggingFace, EncodeAndDecode) {
  const std::string model = GetParam();
  LOAD_TOKENIZER_OR_SKIP(tok, model);
  std::ifstream f(kSource + "/tests/data/tokenizer/" + model + ".json");
  ASSERT_TRUE(f);
  const auto fixture = nlohmann::json::parse(f);

  int n = 0;
  for (const auto& c : fixture.at("cases")) {
    const std::string text = c.at("text");
    const auto want = c.at("ids").get<std::vector<int32_t>>();
    const auto got = tok.encode(text);
    EXPECT_EQ(got, want) << "text: \"" << text << "\"\n  got:  " << show_ids(got)
                         << "\n  want: " << show_ids(want);
    EXPECT_EQ(tok.decode(want), text);
    ++n;
  }
  std::printf("    %d cases\n", n);
}

INSTANTIATE_TEST_SUITE_P(Tokenizer, MatchesHuggingFace,
                         ::testing::Values("gpt2", "llama3.2-1b", "qwen2.5-1.5b"),
                         [](const auto& info) {
                           std::string s = info.param;
                           for (char& ch : s) if (!isalnum(static_cast<unsigned char>(ch))) ch = '_';
                           return s;
                         });

// Rendering matches transformers' apply_chat_template (date pinned to the fixture's).
TEST(ChatTemplate, MatchesHuggingFace) {
  for (const std::string model : {"qwen2.5-1.5b", "llama3.2-1b"}) {
    SCOPED_TRACE(model);
    const std::string config = kSource + "/models/" + model + "/tokenizer_config.json";
    if (!fs::exists(config)) GTEST_SKIP() << config << " not found";
    ChatTemplate tmpl = ChatTemplate::from_tokenizer_config(config);
    tmpl.set_date_string("26 Jul 2024");
    std::printf("    %s: %s\n", model.c_str(), tmpl.format_name());

    std::ifstream f(kSource + "/tests/data/tokenizer/" + model + ".json");
    ASSERT_TRUE(f);
    const auto chats = nlohmann::json::parse(f).at("chats");
    ASSERT_FALSE(chats.empty());
    for (const auto& c : chats) {
      std::vector<ChatMessage> messages;
      for (const auto& m : c.at("messages")) messages.push_back({m.at("role"), m.at("content")});
      EXPECT_EQ(tmpl.render(messages, c.at("add_generation_prompt")), c.at("text").get<std::string>());
    }
  }
}

TEST(ChatTemplate, RawWithoutTemplate) {
  const auto tmpl = ChatTemplate::from_tokenizer_config(kSource + "/models/gpt2/tokenizer_config.json");
  EXPECT_TRUE(tmpl.is_raw());
  EXPECT_EQ(tmpl.render({{"user", "The capital"}, {"assistant", " of France"}}), "The capital of France");
}

TEST(ChatTemplate, RejectsUnknownRole) {
  const auto tmpl = ChatTemplate::from_template_text("<|im_start|>", "");
  EXPECT_THROW(tmpl.render({{"tool", "x"}}), std::invalid_argument);
}

TEST(Tokenizer, SpecialTokenIds) {
  LOAD_TOKENIZER_OR_SKIP(gpt2, "gpt2");
  EXPECT_EQ(gpt2.vocab_size(), 50257);
  EXPECT_EQ(gpt2.bos_id(), 50256);
  EXPECT_EQ(gpt2.eos_ids(), std::vector<int32_t>{50256});
  EXPECT_EQ(gpt2.token_id("<|endoftext|>"), 50256);
  EXPECT_EQ(gpt2.token_id("Ġthe"), 262);
  EXPECT_EQ(gpt2.token_id("no such token"), -1);
  EXPECT_EQ(gpt2.encode("<|endoftext|>"), std::vector<int32_t>{50256});

  LOAD_TOKENIZER_OR_SKIP(llama, "llama3.2-1b");
  EXPECT_EQ(llama.bos_id(), 128000);
  EXPECT_EQ(llama.eos_ids(), (std::vector<int32_t>{128001, 128008, 128009}));
  EXPECT_TRUE(llama.is_eos(llama.token_id("<|eot_id|>")));
}

TEST(Tokenizer, EncodesLargeTextQuickly) {
  LOAD_TOKENIZER_OR_SKIP(tok, "gpt2");
  std::string text;
  while (text.size() < (1u << 20)) {
    text += "The quick brown fox jumps over the lazy dog. 1234567 naïve 你好 😀\n\t  ";
  }
  text += std::string(100000, 'a');  // one huge piece: must not be quadratic
  const auto t0 = std::chrono::steady_clock::now();
  const auto ids = tok.encode(text);
  const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("    %.1f MB -> %zu tokens in %.0f ms\n", text.size() / 1e6, ids.size(), ms);
  EXPECT_EQ(tok.decode(ids), text);
  EXPECT_LT(ms, 5000);
}

TEST(IncrementalDecoder, HoldsBackPartialUtf8) {
  LOAD_TOKENIZER_OR_SKIP(tok, "gpt2");
  // GPT-2 splits most emoji and CJK characters across several byte tokens.
  const std::string text = "naïve 你好世界 😀👍🏽 done";
  IncrementalDecoder dec(tok);
  std::string streamed;
  for (int32_t id : tok.encode(text)) {
    const std::string piece = dec.push(id);
    EXPECT_TRUE(valid_utf8(piece)) << "partial character leaked";
    streamed += piece;
  }
  streamed += dec.flush();
  EXPECT_EQ(streamed, text);
}

TEST(IncrementalDecoder, ReplacesMalformedBytes) {
  LOAD_TOKENIZER_OR_SKIP(tok, "gpt2");
  const auto byte_token = [&](unsigned char b) { return tok.encode(std::string(1, char(b)))[0]; };
  const auto ok = tok.encode("ok");

  IncrementalDecoder dec(tok);
  // A lone continuation byte is malformed immediately.
  EXPECT_EQ(dec.push(byte_token(0x80)), "\xEF\xBF\xBD");
  // A lead byte waits for its continuation...
  EXPECT_EQ(dec.push(byte_token(0xE4)), "");
  // ...and is replaced once the next byte shows it never completes.
  std::string out = dec.push(ok[0]);
  for (size_t i = 1; i < ok.size(); ++i) out += dec.push(ok[i]);
  EXPECT_EQ(out, "\xEF\xBF\xBDok");
  // An incomplete sequence at the end is replaced on flush.
  EXPECT_EQ(dec.push(byte_token(0xF0)), "");
  EXPECT_EQ(dec.flush(), "\xEF\xBF\xBD");
}

TEST(Unicode, CompleteUtf8Prefix) {
  using unicode::complete_utf8_prefix;
  EXPECT_EQ(complete_utf8_prefix(""), 0u);
  EXPECT_EQ(complete_utf8_prefix("abc"), 3u);
  EXPECT_EQ(complete_utf8_prefix("a\xE4\xBD"), 1u);      // 2 of 3 bytes of 你
  EXPECT_EQ(complete_utf8_prefix("a\xE4\xBD\xA0"), 4u);  // complete 你
  EXPECT_EQ(complete_utf8_prefix("a\xF0\x9F\x98"), 1u);  // 3 of 4 bytes of 😀
  EXPECT_EQ(complete_utf8_prefix("a\xE0\x80"), 3u);      // overlong start: malformed, not held
  EXPECT_EQ(complete_utf8_prefix("a\x80"), 2u);          // stray continuation: malformed
}

TEST(Unicode, Classes) {
  EXPECT_TRUE(unicode::is_letter(U'a'));
  EXPECT_TRUE(unicode::is_letter(U'ж'));
  EXPECT_TRUE(unicode::is_letter(U'中'));
  EXPECT_FALSE(unicode::is_letter(U'́'));  // combining acute: Mn, not L
  EXPECT_TRUE(unicode::is_number(U'7'));
  EXPECT_TRUE(unicode::is_number(U'²'));
  EXPECT_TRUE(unicode::is_number(U'Ⅻ'));
  EXPECT_TRUE(unicode::is_whitespace(U'　'));
  EXPECT_FALSE(unicode::is_whitespace(U'​'));  // zero-width space is not White_Space
}
