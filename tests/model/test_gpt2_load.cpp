// Loads the real openai-community/gpt2 checkpoint onto the GPU and checks the weight mapping.
// Looks in $OGRAPHY_GPT2_DIR, else <repo>/models/gpt2; skipped if neither exists.

#include <gtest/gtest.h>

#include <cuda_fp16.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/memory.h"
#include "model/config.h"
#include "model/safetensors.h"
#include "model/weight_map.h"

using namespace ography;
namespace fs = std::filesystem;

namespace {

std::string gpt2_dir() {
  if (const char* env = std::getenv("OGRAPHY_GPT2_DIR")) return env;
  return std::string(OGRAPHY_SOURCE_DIR) + "/models/gpt2";
}

std::vector<__half> download(const Tensor& t) {
  std::vector<__half> out(t.numel());
  CUDA_CHECK(cudaMemcpy(out.data(), t.data(), t.nbytes(), cudaMemcpyDeviceToHost));
  return out;
}

// fp32 -> fp16 -> fp32, i.e. what the GPU copy should hold.
float rounded(float v) { return __half2float(__float2half(v)); }

class Gpt2Load : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!fs::exists(gpt2_dir() + "/model.safetensors")) return;
    config_ = new ModelConfig(ModelConfig::from_file(gpt2_dir() + "/config.json"));
    ckpt_ = new SafetensorsModel(gpt2_dir());
    arena_ = new DeviceArena(size_t{512} << 20);
    weights_ = new Gpt2Weights(map_gpt2_weights(*ckpt_, *config_, *arena_));
  }
  static void TearDownTestSuite() {
    delete weights_;
    delete arena_;
    delete ckpt_;
    delete config_;
  }
  void SetUp() override {
    if (weights_ == nullptr) GTEST_SKIP() << "GPT-2 checkpoint not found at " << gpt2_dir();
  }

  static inline ModelConfig* config_ = nullptr;
  static inline SafetensorsModel* ckpt_ = nullptr;
  static inline DeviceArena* arena_ = nullptr;
  static inline Gpt2Weights* weights_ = nullptr;
};

}  // namespace

TEST_F(Gpt2Load, ShapesAndDtypes) {
  const auto& w = *weights_;
  EXPECT_EQ(w.layers.size(), 12u);
  EXPECT_EQ(w.wte.shape(), (std::vector<int64_t>{50257, 768}));
  EXPECT_EQ(w.wpe.shape(), (std::vector<int64_t>{1024, 768}));
  EXPECT_EQ(w.layers[0].wqkv.weight.shape(), (std::vector<int64_t>{2304, 768}));
  EXPECT_EQ(w.layers[0].w_fc.weight.shape(), (std::vector<int64_t>{3072, 768}));
  EXPECT_EQ(w.layers[0].w_proj.weight.shape(), (std::vector<int64_t>{768, 3072}));
  EXPECT_EQ(w.layers[11].wqkv.weight.dtype(), DType::F16);
  EXPECT_EQ(w.lm_head.weight.data(), w.wte.data());  // tied
}

TEST_F(Gpt2Load, FitsInExpectedMemory) {
  // ~124M params * 2 bytes, minus nothing for lm_head (tied) and the skipped mask buffers.
  const double mib = static_cast<double>(arena_->used()) / (1 << 20);
  EXPECT_GT(mib, 230.0);
  EXPECT_LT(mib, 250.0);
}

TEST_F(Gpt2Load, Conv1DWeightsAreTransposed) {
  for (int layer : {0, 11}) {
    const std::string name = "h." + std::to_string(layer) + ".attn.c_attn.weight";
    const Tensor src = ckpt_->get(name);  // [768 in, 2304 out], fp32
    const auto gpu = download(weights_->layers[layer].wqkv.weight);  // [2304 out, 768 in]
    const float* s = src.data_as<float>();
    for (auto [in, out] : {std::pair{0, 0}, {1, 0}, {0, 1}, {767, 2303}, {123, 1800}, {500, 7}}) {
      EXPECT_EQ(__half2float(gpu[static_cast<size_t>(out) * 768 + in]),
                rounded(s[static_cast<size_t>(in) * 2304 + out]))
          << name << " in=" << in << " out=" << out;
    }
  }
}

TEST_F(Gpt2Load, VectorsAndEmbeddingsCopiedVerbatim) {
  const Tensor src_bias = ckpt_->get("h.3.mlp.c_fc.bias");
  const auto gpu_bias = download(weights_->layers[3].b_fc);
  for (size_t i = 0; i < gpu_bias.size(); i += 97) {
    EXPECT_EQ(__half2float(gpu_bias[i]), rounded(src_bias.data_as<float>()[i])) << i;
  }

  const Tensor src_wte = ckpt_->get("wte.weight");
  const auto gpu_wte = download(weights_->wte);
  for (size_t i : {size_t{0}, size_t{768} * 50256 + 767, size_t{12345678}}) {
    EXPECT_EQ(__half2float(gpu_wte[i]), rounded(src_wte.data_as<float>()[i])) << i;
  }
}
