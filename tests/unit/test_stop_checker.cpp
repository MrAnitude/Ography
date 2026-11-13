#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "engine/stop_checker.h"

using namespace ography;

namespace {

constexpr int32_t kEos = 99;
constexpr int32_t kTok = 1;  // any ordinary token

SamplingParams params(std::vector<std::string> stop, int max_tokens = 100) {
  SamplingParams p;
  p.stop = std::move(stop);
  p.max_tokens = max_tokens;
  return p;
}

// Feeds pieces one token each; returns everything emitted and the finish reason (if any).
std::pair<std::string, std::optional<FinishReason>> run(StopChecker& sc,
                                                        const std::vector<std::string>& pieces) {
  std::string out;
  int n = 0;
  for (const auto& p : pieces) {
    auto r = sc.push(kTok, p, ++n);
    out += r.emit;
    if (r.finish) return {out, r.finish};
  }
  return {out, std::nullopt};
}

}  // namespace

TEST(StopChecker, StopStringAcrossTokens) {
  StopChecker sc(params({"\nUser:"}), {kEos});
  // "\nUs" might start the stop string, so it is held back; the match is never emitted.
  EXPECT_EQ(sc.push(kTok, "Hello", 1).emit, "Hello");
  auto r = sc.push(kTok, " there\nUs", 2);
  EXPECT_EQ(r.emit, " there");
  EXPECT_FALSE(r.finish);
  r = sc.push(kTok, "er: more", 3);
  EXPECT_EQ(r.emit, "");
  EXPECT_EQ(r.finish, FinishReason::Stop);
}

TEST(StopChecker, HeldTextReleasedWhenRuledOut) {
  StopChecker sc(params({"STOP"}), {kEos});
  auto [out, finish] = run(sc, {"a S", "T", "OP", "!"});
  EXPECT_EQ(out, "a ");
  EXPECT_EQ(finish, FinishReason::Stop);

  StopChecker sc2(params({"STOP"}), {kEos});
  auto [out2, finish2] = run(sc2, {"a S", "T", "x"});
  EXPECT_EQ(out2, "a STx");
  EXPECT_FALSE(finish2);
}

TEST(StopChecker, EarliestOfSeveralStopStrings) {
  StopChecker sc(params({"world", "lo w"}), {kEos});
  auto [out, finish] = run(sc, {"hello world"});
  EXPECT_EQ(out, "hel");
  EXPECT_EQ(finish, FinishReason::Stop);
}

TEST(StopChecker, StopTokenFinishesAndFlushes) {
  StopChecker sc(params({"END"}), {kEos});
  EXPECT_EQ(sc.push(kTok, "abc E", 1).emit, "abc ");  // "E" held back
  auto r = sc.push(kEos, "", 2);
  EXPECT_EQ(r.emit, "E");
  EXPECT_EQ(r.finish, FinishReason::Stop);
}

TEST(StopChecker, RequestStopTokenIds) {
  SamplingParams p = params({});
  p.stop_token_ids = {7};
  StopChecker sc(p, {kEos});
  EXPECT_TRUE(sc.is_stop_token(7));
  EXPECT_TRUE(sc.is_stop_token(kEos));
  EXPECT_FALSE(sc.is_stop_token(kTok));
  EXPECT_EQ(sc.push(7, "", 1).finish, FinishReason::Stop);
}

TEST(StopChecker, MaxTokensFinishesWithLength) {
  StopChecker sc(params({"zzz"}, 3), {kEos});
  auto [out, finish] = run(sc, {"a", "b", "cz", "d"});
  EXPECT_EQ(out, "abcz");  // held-back "z" is flushed at the limit
  EXPECT_EQ(finish, FinishReason::Length);
}

TEST(StopChecker, EmptyStopStringIgnored) {
  StopChecker sc(params({""}), {kEos});
  auto [out, finish] = run(sc, {"abc"});
  EXPECT_EQ(out, "abc");
  EXPECT_FALSE(finish);
}
