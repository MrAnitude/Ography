#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "model/rope.h"

using namespace ography;

// Llama 3.2's rope_scaling (factor 32, low 1, high 4, original 8192, theta 5e5), compared with
// transformers' ROPE_INIT_FUNCTIONS["llama3"].
TEST(Rope, Llama3ScalingMatchesHF) {
  ModelConfig c;
  c.head_dim = 64;
  c.rope_theta = 500000.0f;
  c.rope_scaling = {"llama3", 32.0f, 1.0f, 4.0f, 8192};
  const std::vector<float> hf = {
      1.f, 0.663601279f, 0.440366626f, 0.292227834f, 0.193922758f, 0.128687382f,
      0.0853971019f, 0.0566696189f, 0.0376060307f, 0.0249554086f, 0.0165604409f,
      0.0109895291f, 0.00729266508f, 0.00483942125f, 0.00321144611f, 0.00129054801f,
      0.000429556705f, 9.70828623e-05f, 1.94616387e-05f, 1.29147675e-05f, 8.57025589e-06f,
      5.68723226e-06f, 3.77405445e-06f, 2.50446715e-06f, 1.66196742e-06f, 1.10288363e-06f,
      7.31874934e-07f, 4.85673127e-07f, 3.22293289e-07f, 2.1387423e-07f, 1.41927202e-07f,
      9.41830649e-08f};
  const auto got = rope_inv_freq(c);
  ASSERT_EQ(got.size(), hf.size());
  for (size_t i = 0; i < hf.size(); ++i) EXPECT_NEAR(got[i], hf[i], hf[i] * 1e-6f) << i;
}

TEST(Rope, DefaultAndLinear) {
  ModelConfig c;
  c.head_dim = 128;
  c.rope_theta = 1000000.0f;
  auto inv = rope_inv_freq(c);
  ASSERT_EQ(inv.size(), 64u);
  EXPECT_FLOAT_EQ(inv[0], 1.0f);
  EXPECT_NEAR(inv[63], 1.0f / std::pow(1e6f, 126.0f / 128.0f), 1e-12f);
  c.rope_scaling.type = "linear";
  c.rope_scaling.factor = 4.0f;
  EXPECT_FLOAT_EQ(rope_inv_freq(c)[1], inv[1] / 4.0f);
  c.rope_scaling.type = "yarn";
  EXPECT_THROW(rope_inv_freq(c), std::runtime_error);
}
