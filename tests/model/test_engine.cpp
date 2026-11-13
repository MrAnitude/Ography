// Engine::generate end to end on GPT-2: greedy output vs HF, streaming, stop handling,
// cancellation, seeded sampling, penalties and chunked prefill. Skipped if the checkpoint
// isn't present.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "engine/engine.h"
#include "model/safetensors.h"
#include "tokenizer/tokenizer.h"

using namespace ography;
namespace fs = std::filesystem;

namespace {

std::string gpt2_dir() {
  if (const char* env = std::getenv("OGRAPHY_GPT2_DIR")) return env;
  return std::string(OGRAPHY_SOURCE_DIR) + "/models/gpt2";
}

// HF transformers' greedy continuation of "The capital of France is" (tools/dump_golden.py).
const std::vector<int32_t> kHfGreedy = {262,  3139, 286,  262, 4141, 2066, 11,  290,
                                        262,  3139, 286,  262, 4141, 2066, 318, 262,
                                        3139, 286,  262,  4141, 2066, 13,   198, 198,
                                        464,  4141, 2066, 318, 262,  3139, 286, 262};

struct GenRun {
  std::vector<int32_t> ids;
  std::string text;
  GenerationResult result;
};

GenRun generate(Engine& engine, const std::vector<int32_t>& prompt, SamplingParams sp,
             int cancel_after = -1) {
  GenRun run;
  GenerationRequest req{prompt, sp};
  run.result = engine.generate(req, [&](const TokenEvent& ev) {
    run.ids.push_back(ev.token_id);
    run.text += ev.text;
    return cancel_after < 0 || static_cast<int>(run.ids.size()) < cancel_after;
  });
  return run;
}

SamplingParams greedy(int max_tokens) {
  SamplingParams sp;
  sp.temperature = 0;
  sp.max_tokens = max_tokens;
  return sp;
}

class EngineTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!fs::exists(gpt2_dir() + "/model.safetensors")) return;
    EngineConfig cfg;
    cfg.model_dir = gpt2_dir();
    engine_ = std::make_unique<Engine>(cfg);
  }
  static void TearDownTestSuite() { engine_.reset(); }
  void SetUp() override {
    if (!engine_) GTEST_SKIP() << "GPT-2 checkpoint not found at " << gpt2_dir();
    prompt_ = engine_->tokenizer().encode("The capital of France is");
  }

  static std::unique_ptr<Engine> engine_;
  std::vector<int32_t> prompt_;
};

std::unique_ptr<Engine> EngineTest::engine_;

}  // namespace

TEST_F(EngineTest, GreedyMatchesHF) {
  ASSERT_EQ(prompt_, (std::vector<int32_t>{464, 3139, 286, 4881, 318}));
  const GenRun run = generate(*engine_, prompt_, greedy(32));
  EXPECT_EQ(run.ids, kHfGreedy);
  EXPECT_EQ(run.result.finish_reason, FinishReason::Length);
  EXPECT_EQ(run.result.prompt_tokens, 5);
  EXPECT_EQ(run.result.completion_tokens, 32);
  // Streamed text is exactly the decoded completion.
  EXPECT_EQ(run.text, engine_->tokenizer().decode(run.ids));
  std::printf("    \"%s\"\n    TTFT %.1f ms, %.0f tok/s\n", run.text.c_str(),
              run.result.time_to_first_token_ms, run.result.tokens_per_second);
}

TEST_F(EngineTest, StopStringIsNotEmitted) {
  SamplingParams sp = greedy(32);
  sp.stop = {" French"};
  const GenRun run = generate(*engine_, prompt_, sp);
  EXPECT_EQ(run.text, " the capital of the");
  EXPECT_EQ(run.result.finish_reason, FinishReason::Stop);
}

TEST_F(EngineTest, StopTokenIdFinishes) {
  SamplingParams sp = greedy(32);
  sp.stop_token_ids = {kHfGreedy[4]};  // " French"
  const GenRun run = generate(*engine_, prompt_, sp);
  EXPECT_EQ(run.ids, std::vector<int32_t>(kHfGreedy.begin(), kHfGreedy.begin() + 5));
  EXPECT_EQ(run.text, " the capital of the");  // the stop token's text is not shown
  EXPECT_EQ(run.result.finish_reason, FinishReason::Stop);
}

TEST_F(EngineTest, CallbackCancels) {
  const GenRun run = generate(*engine_, prompt_, greedy(32), /*cancel_after=*/3);
  EXPECT_EQ(run.ids.size(), 3u);
  EXPECT_EQ(run.result.finish_reason, FinishReason::Cancelled);
  // The engine is reusable afterwards.
  EXPECT_EQ(generate(*engine_, prompt_, greedy(32)).ids, kHfGreedy);
}

TEST_F(EngineTest, SeededSamplingIsReproducible) {
  SamplingParams sp;
  sp.temperature = 0.9f;
  sp.top_p = 0.95f;
  sp.max_tokens = 24;
  sp.seed = 1234;
  const GenRun a = generate(*engine_, prompt_, sp);
  EXPECT_EQ(a.ids, generate(*engine_, prompt_, sp).ids);
  sp.seed = 999;
  EXPECT_NE(a.ids, generate(*engine_, prompt_, sp).ids);
  std::printf("    seed 1234: \"%s\"\n", a.text.c_str());
}

TEST_F(EngineTest, RepetitionPenaltyBreaksLoops) {
  // Plain greedy repeats "the capital of the French Republic"; the penalty must change that.
  SamplingParams sp = greedy(32);
  sp.repetition_penalty = 1.3f;
  const GenRun run = generate(*engine_, prompt_, sp);
  EXPECT_NE(run.ids, kHfGreedy);
  EXPECT_EQ(run.ids.size(), 32u);
  std::printf("    penalised: \"%s\"\n", run.text.c_str());
}

TEST_F(EngineTest, MaxTokensClampedToContext) {
  // 1020 prompt tokens in a 1024-token context leave room for exactly 5 generated tokens.
  std::vector<int32_t> prompt(1020, 464);
  const GenRun run = generate(*engine_, prompt, greedy(100));
  EXPECT_EQ(run.ids.size(), 5u);
  EXPECT_EQ(run.result.finish_reason, FinishReason::Length);
}

TEST_F(EngineTest, RejectsBadPrompts) {
  EXPECT_THROW(generate(*engine_, {}, greedy(4)), std::invalid_argument);
  EXPECT_THROW(generate(*engine_, std::vector<int32_t>(1024, 464), greedy(4)),
               std::invalid_argument);
  EXPECT_THROW(generate(*engine_, {50257}, greedy(4)), std::invalid_argument);
}

// A 460-token prompt with 64-token prefill chunks must reproduce HF's greedy continuation.
TEST(EngineChunkedPrefill, MatchesHF) {
  const std::string golden = std::string(OGRAPHY_SOURCE_DIR) + "/golden/gpt2/long.safetensors";
  if (!fs::exists(gpt2_dir() + "/model.safetensors") || !fs::exists(golden)) {
    GTEST_SKIP() << "needs models/gpt2 and golden/gpt2 (tools/dump_golden.py)";
  }
  SafetensorsFile f(golden);
  auto vec = [&](const char* name) {
    const Tensor t = f.get(name);
    return std::vector<int32_t>(t.data_as<int32_t>(), t.data_as<int32_t>() + t.numel());
  };
  EngineConfig cfg;
  cfg.model_dir = gpt2_dir();
  cfg.prefill_chunk = 64;
  Engine engine(cfg);
  const auto want = vec("greedy_ids");
  const GenRun run = generate(engine, vec("input_ids"), greedy(static_cast<int>(want.size())));
  EXPECT_EQ(run.ids, want);
  std::printf("    460-token prompt: TTFT %.1f ms, %.0f tok/s\n",
              run.result.time_to_first_token_ms, run.result.tokens_per_second);
}
