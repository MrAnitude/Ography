// Sampling and penalty kernels vs CPU references. Stochastic modes are checked
// statistically: thousands of draws (one Philox subsequence each) against the exact
// expected distribution after filtering.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "kernels/kernels.h"
#include "test_util.h"

using namespace ography;
using namespace ography::test;
using ography::kernels::SamplingKernelParams;

namespace {

int cpu_argmax(const std::vector<float>& v) {
  return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

// Expected sampling distribution (HF order: temperature, top-k, top-p, min-p).
std::vector<double> expected_probs(const std::vector<float>& logits, const SamplingKernelParams& p) {
  const size_t n = logits.size();
  const float m = *std::max_element(logits.begin(), logits.end());
  std::vector<double> prob(n);
  for (size_t i = 0; i < n; ++i) prob[i] = std::exp((logits[i] - m) / p.temperature);
  std::vector<double> sorted = prob;
  std::sort(sorted.rbegin(), sorted.rend());

  double threshold = 0;
  if (p.top_k > 0 && size_t(p.top_k) < n) threshold = sorted[p.top_k - 1];
  if (p.top_p < 1.f) {
    double kept = 0;
    for (double x : sorted) if (x >= threshold) kept += x;
    double acc = 0;
    for (double x : sorted) {
      acc += x;
      if (acc >= p.top_p * kept) {
        threshold = std::max(threshold, x);
        break;
      }
    }
  }
  threshold = std::max(threshold, double(p.min_p));
  double total = 0;
  for (auto& x : prob) total += (x = x >= threshold ? x : 0);
  for (auto& x : prob) x /= total;
  return prob;
}

class Sampling : public ::testing::Test {
 protected:
  DeviceArena arena{size_t{64} << 20};

  // Draws n tokens with steps 0..n-1.
  std::vector<int32_t> draw(const std::vector<float>& logits, SamplingKernelParams p, int n) {
    Tensor d_logits = to_device(arena, logits, DType::F32, {int64_t(logits.size())});
    Tensor scratch = arena.allocate(DType::F32, {int64_t(logits.size())});
    Tensor out = arena.allocate(DType::I32, {n});
    for (int i = 0; i < n; ++i) {
      p.step = i;
      kernels::sample(d_logits, p, out.data_as<int32_t>() + i, scratch, nullptr);
    }
    std::vector<int32_t> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), out.data(), n * 4, cudaMemcpyDeviceToHost));
    return h;
  }

  // Draws many tokens and checks every frequency against the expected distribution.
  void check_distribution(const std::vector<float>& logits, const SamplingKernelParams& p) {
    const int n = 20000;
    const auto want = expected_probs(logits, p);
    const auto tokens = draw(logits, p, n);
    std::vector<int> hist(logits.size());
    for (int32_t t : tokens) {
      ASSERT_GE(t, 0);
      ASSERT_LT(t, int(logits.size()));
      ++hist[t];
    }
    for (size_t i = 0; i < logits.size(); ++i) {
      if (want[i] == 0) {
        EXPECT_EQ(hist[i], 0) << "token " << i << " should have been filtered out";
        continue;
      }
      const double sigma = std::sqrt(n * want[i] * (1 - want[i]));
      EXPECT_NEAR(hist[i], n * want[i], 5 * sigma + 1) << "token " << i;
    }
  }

  static std::vector<float> small_logits() {
    return {2.0f, 1.5f, 1.0f, 0.5f, 0.0f, -0.5f, -1.0f, 3.0f, 2.5f, -2.0f};
  }
};

}  // namespace

TEST_F(Sampling, GreedyMatchesArgmax) {
  for (int n : {7, 1000, 50257, 151936}) {
    const auto logits = random_f16_values(n, n, -20.f, 20.f);
    SamplingKernelParams p;
    p.temperature = 0;
    EXPECT_EQ(draw(logits, p, 1)[0], cpu_argmax(logits)) << "vocab " << n;
  }
}

TEST_F(Sampling, GreedyTiesPickLowestId) {
  std::vector<float> logits(5000, 0.f);
  logits[4321] = logits[1234] = logits[3000] = 5.f;
  SamplingKernelParams p;
  p.temperature = 0;
  EXPECT_EQ(draw(logits, p, 1)[0], 1234);
  p.temperature = 1;
  p.top_k = 1;  // top_k = 1 is greedy too
  EXPECT_EQ(draw(logits, p, 1)[0], 1234);
}

TEST_F(Sampling, Temperature) {
  for (float t : {1.0f, 0.5f, 2.0f}) {
    SCOPED_TRACE(t);
    SamplingKernelParams p;
    p.temperature = t;
    p.seed = 7;
    check_distribution(small_logits(), p);
  }
}

TEST_F(Sampling, TopK) {
  SamplingKernelParams p;
  p.top_k = 3;
  p.seed = 1;
  check_distribution(small_logits(), p);
}

TEST_F(Sampling, TopP) {
  for (float top_p : {0.5f, 0.8f, 0.95f}) {
    SCOPED_TRACE(top_p);
    SamplingKernelParams p;
    p.top_p = top_p;
    p.seed = 2;
    check_distribution(small_logits(), p);
  }
}

TEST_F(Sampling, MinP) {
  SamplingKernelParams p;
  p.min_p = 0.2f;
  p.seed = 3;
  check_distribution(small_logits(), p);
}

TEST_F(Sampling, CombinedFilters) {
  SamplingKernelParams p;
  p.temperature = 0.8f;
  p.top_k = 6;
  p.top_p = 0.9f;
  p.min_p = 0.05f;
  p.seed = 4;
  check_distribution(small_logits(), p);
}

TEST_F(Sampling, LargeVocabStaysInTopK) {
  const auto logits = random_f16_values(151936, 11, -10.f, 10.f);
  SamplingKernelParams p;
  p.top_k = 40;
  p.top_p = 0.9f;
  p.seed = 5;
  std::vector<float> sorted = logits;
  std::sort(sorted.rbegin(), sorted.rend());
  for (int32_t t : draw(logits, p, 500)) EXPECT_GE(logits[t], sorted[39]);
}

TEST_F(Sampling, SeedIsReproducible) {
  const auto logits = random_f16_values(1000, 3, -2.f, 2.f);
  SamplingKernelParams p;
  p.seed = 42;
  const auto a = draw(logits, p, 64);
  EXPECT_EQ(a, draw(logits, p, 64));
  p.seed = 43;
  EXPECT_NE(a, draw(logits, p, 64));
}

TEST(Penalties, MatchCpuReference) {
  DeviceArena arena(size_t{16} << 20);
  const int n = 3000;
  const auto logits = random_f16_values(n, 9, -5.f, 5.f);
  std::vector<int32_t> counts(n, 0), prompt_ids = {5, 17, 17, 2999, 100};
  std::vector<int8_t> prompt(n, 0);
  std::mt19937 rng(1);
  for (int k = 0; k < 200; ++k) counts[rng() % n] += 1 + rng() % 3;
  for (int id : prompt_ids) prompt[id] = 1;

  const float rep = 1.3f, pres = 0.4f, freq = 0.25f;
  std::vector<float> want = logits;
  for (int i = 0; i < n; ++i) {
    if (counts[i] > 0 || prompt[i]) want[i] = want[i] > 0 ? want[i] / rep : want[i] * rep;
    if (counts[i] > 0) want[i] -= pres + freq * counts[i];
  }

  Tensor d_logits = to_device(arena, logits, DType::F32, {n});
  Tensor d_counts = to_device_i32(arena, counts);
  Tensor d_mask = arena.allocate(DType::I8, {n});
  CUDA_CHECK(cudaMemset(d_mask.data(), 0, n));
  kernels::mark_tokens(to_device_i32(arena, prompt_ids), d_mask, nullptr);
  kernels::apply_penalties(d_logits, d_mask, d_counts, rep, pres, freq, nullptr);
  EXPECT_LT(max_abs_diff(to_host(d_logits), want), 1e-5f);

  // count_token reads the token id from device memory.
  Tensor tok = to_device_i32(arena, {17});
  kernels::count_token(tok.data_as<int32_t>(), d_counts, nullptr);
  kernels::count_token(tok.data_as<int32_t>(), d_counts, nullptr);
  int32_t c17 = 0;
  CUDA_CHECK(cudaMemcpy(&c17, d_counts.data_as<int32_t>() + 17, 4, cudaMemcpyDeviceToHost));
  EXPECT_EQ(c17, counts[17] + 2);
}
