// End-to-end GPT-2 forward pass on the GPU vs a straightforward fp32 CPU implementation
// built directly from the checkpoint. Skipped if the checkpoint isn't present.
//
// This checks the kernels compose correctly (layout, transposes, residual wiring, KV cache
// offsets). Agreement with HF transformers itself is the job of tests/golden.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/memory.h"
#include "kv_cache/kv_cache.h"
#include "model/arch/gpt2.h"
#include "model/registry.h"
#include "model/safetensors.h"

using namespace ography;
namespace fs = std::filesystem;

namespace {

std::string gpt2_dir() {
  if (const char* env = std::getenv("OGRAPHY_GPT2_DIR")) return env;
  return std::string(OGRAPHY_SOURCE_DIR) + "/models/gpt2";
}

// "The capital of France is" in GPT-2's BPE.
const std::vector<int32_t> kPrompt = {464, 3139, 286, 4881, 318};

// ---- CPU fp32 reference -------------------------------------------------------------------

class CpuGpt2 {
 public:
  CpuGpt2(const SafetensorsModel& ckpt, const ModelConfig& c) : ckpt_(ckpt), c_(c) {}

  // Logits of the last token after running the whole sequence.
  std::vector<float> last_logits(const std::vector<int32_t>& ids) const {
    const int T = static_cast<int>(ids.size());
    const int H = c_.hidden_size;
    const float* wte = f("wte.weight");
    const float* wpe = f("wpe.weight");
    std::vector<float> h(static_cast<size_t>(T) * H);
    for (int t = 0; t < T; ++t)
      for (int i = 0; i < H; ++i) h[t * H + i] = wte[ids[t] * H + i] + wpe[t * H + i];

    for (int l = 0; l < c_.num_layers; ++l) {
      const std::string p = "h." + std::to_string(l) + ".";
      auto x = layernorm(h, T, p + "ln_1");
      auto qkv = linear(x, T, H, 3 * H, p + "attn.c_attn");
      auto a = attention(qkv, T);
      auto o = linear(a, T, H, H, p + "attn.c_proj");
      for (size_t i = 0; i < h.size(); ++i) h[i] += o[i];

      x = layernorm(h, T, p + "ln_2");
      auto fc = linear(x, T, H, c_.intermediate_size, p + "mlp.c_fc");
      for (auto& v : fc) v = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
      auto m = linear(fc, T, c_.intermediate_size, H, p + "mlp.c_proj");
      for (size_t i = 0; i < h.size(); ++i) h[i] += m[i];
    }
    auto x = layernorm(h, T, "ln_f");
    std::vector<float> logits(c_.vocab_size);
    const float* last = &x[static_cast<size_t>(T - 1) * H];
    for (int v = 0; v < c_.vocab_size; ++v) {
      float acc = 0;
      for (int i = 0; i < H; ++i) acc += last[i] * wte[static_cast<size_t>(v) * H + i];
      logits[v] = acc;
    }
    return logits;
  }

 private:
  const float* f(const std::string& name) const { return ckpt_.get(name).data_as<const float>(); }

  std::vector<float> layernorm(const std::vector<float>& x, int T, const std::string& p) const {
    const int H = c_.hidden_size;
    const float* w = f(p + ".weight");
    const float* b = f(p + ".bias");
    std::vector<float> out(x.size());
    for (int t = 0; t < T; ++t) {
      const float* r = &x[static_cast<size_t>(t) * H];
      double mean = 0, var = 0;
      for (int i = 0; i < H; ++i) mean += r[i];
      mean /= H;
      for (int i = 0; i < H; ++i) var += (r[i] - mean) * (r[i] - mean);
      const double rstd = 1.0 / std::sqrt(var / H + c_.norm_eps);
      for (int i = 0; i < H; ++i) out[static_cast<size_t>(t) * H + i] = float((r[i] - mean) * rstd) * w[i] + b[i];
    }
    return out;
  }

  // Conv1D: weight [in, out].
  std::vector<float> linear(const std::vector<float>& x, int T, int in, int out,
                            const std::string& p) const {
    const float* w = f(p + ".weight");
    const float* b = f(p + ".bias");
    std::vector<float> y(static_cast<size_t>(T) * out);
    for (int t = 0; t < T; ++t) {
      float* yr = &y[static_cast<size_t>(t) * out];
      std::copy(b, b + out, yr);
      for (int i = 0; i < in; ++i) {
        const float xi = x[static_cast<size_t>(t) * in + i];
        const float* wr = w + static_cast<size_t>(i) * out;
        for (int o = 0; o < out; ++o) yr[o] += xi * wr[o];
      }
    }
    return y;
  }

  std::vector<float> attention(const std::vector<float>& qkv, int T) const {
    const int H = c_.hidden_size, nh = c_.num_heads, hd = c_.head_dim;
    const float scale = 1.0f / std::sqrt(float(hd));
    std::vector<float> out(static_cast<size_t>(T) * H);
    for (int h = 0; h < nh; ++h)
      for (int t = 0; t < T; ++t) {
        std::vector<float> s(t + 1);
        float mx = -INFINITY;
        for (int j = 0; j <= t; ++j) {
          float dot = 0;
          for (int d = 0; d < hd; ++d) dot += qkv[t * 3 * H + h * hd + d] * qkv[j * 3 * H + H + h * hd + d];
          mx = std::max(mx, s[j] = dot * scale);
        }
        float sum = 0;
        for (auto& v : s) sum += (v = std::exp(v - mx));
        for (int d = 0; d < hd; ++d) {
          float o = 0;
          for (int j = 0; j <= t; ++j) o += s[j] * qkv[j * 3 * H + 2 * H + h * hd + d];
          out[t * H + h * hd + d] = o / sum;
        }
      }
    return out;
  }

  const SafetensorsModel& ckpt_;
  const ModelConfig& c_;
};

int argmax(const std::vector<float>& v) {
  return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

// ---- Fixture ------------------------------------------------------------------------------

class Gpt2Forward : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!fs::exists(gpt2_dir() + "/model.safetensors")) {
      GTEST_SKIP() << "GPT-2 checkpoint not found at " << gpt2_dir();
    }
    CUBLAS_CHECK(cublasCreate(&cublas_));
    arena_ = std::make_unique<DeviceArena>(size_t{768} << 20);
    model_ = load_model(gpt2_dir(), *arena_, cublas_);
    const auto& c = model_->config();
    KVCacheConfig kv{c.num_layers, c.num_kv_heads, c.head_dim, c.max_position_embeddings,
                     DType::F16, c.num_heads};
    kv_ = std::make_unique<KVCache>(kv, *arena_);
    d_ids_ = arena_->allocate(DType::I32, {64});
    d_pos_ = arena_->allocate(DType::I32, {64});
    d_logits_ = arena_->allocate(DType::F32, {c.vocab_size});
  }
  void TearDown() override {
    if (cublas_) cublasDestroy(cublas_);
  }

  // Runs ids at positions [kv.seq_len(), ...) and returns the last token's logits.
  std::vector<float> run(const std::vector<int32_t>& ids, bool decode) {
    const int n = static_cast<int>(ids.size());
    std::vector<int32_t> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = kv_->seq_len() + i;
    CUDA_CHECK(cudaMemcpy(d_ids_.data(), ids.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_.data(), pos.data(), n * 4, cudaMemcpyHostToDevice));
    ForwardInput in;
    in.token_ids = d_ids_.slice_rows(0, n);
    in.positions = d_pos_.slice_rows(0, n);
    in.n_tokens = n;
    in.start_pos = kv_->seq_len();
    in.is_decode = decode;
    model_->forward(in, *kv_, d_logits_, nullptr);
    kv_->advance(n);
    std::vector<float> out(d_logits_.numel());
    CUDA_CHECK(cudaMemcpy(out.data(), d_logits_.data(), d_logits_.nbytes(), cudaMemcpyDeviceToHost));
    return out;
  }

  std::vector<float> reference(const std::vector<int32_t>& ids) {
    SafetensorsModel ckpt(gpt2_dir());
    return CpuGpt2(ckpt, model_->config()).last_logits(ids);
  }

  static void expect_close(const std::vector<float>& got, const std::vector<float>& want) {
    float max_err = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      ASSERT_TRUE(std::isfinite(got[i])) << "logit " << i;
      max_err = std::max(max_err, std::fabs(got[i] - want[i]));
    }
    // GPT-2 logits are O(100); fp16 weights/activations give errors well under 1.
    EXPECT_LT(max_err, 0.5f);
    EXPECT_EQ(argmax(got), argmax(want));
    std::printf("    max |gpu - cpu| = %.4f, argmax = %d (cpu %d)\n", max_err, argmax(got), argmax(want));
  }

  cublasHandle_t cublas_ = nullptr;
  std::unique_ptr<DeviceArena> arena_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<KVCache> kv_;
  Tensor d_ids_, d_pos_, d_logits_;
};

}  // namespace

TEST_F(Gpt2Forward, PrefillMatchesCpuReference) {
  expect_close(run(kPrompt, /*decode=*/false), reference(kPrompt));
}

TEST_F(Gpt2Forward, PrefillThenDecodeMatchesCpuReference) {
  // Prefill all but the last token, then decode it: exercises KV cache reuse at an offset.
  std::vector<int32_t> head(kPrompt.begin(), kPrompt.end() - 1);
  run(head, /*decode=*/false);
  expect_close(run({kPrompt.back()}, /*decode=*/true), reference(kPrompt));
}

TEST_F(Gpt2Forward, GreedyDecodeIsSelfConsistent) {
  // Generate 8 tokens greedily with decode steps, then check one prefill over the whole
  // sequence predicts the same final token.
  std::vector<int32_t> seq = kPrompt;
  auto logits = run(seq, false);
  for (int i = 0; i < 8; ++i) {
    seq.push_back(argmax(logits));
    logits = run({seq.back()}, true);
  }
  const int decoded_next = argmax(logits);

  kv_->clear();
  EXPECT_EQ(argmax(run(seq, false)), decoded_next);

  std::printf("    greedy ids:");
  for (size_t i = kPrompt.size(); i < seq.size(); ++i) std::printf(" %d", seq[i]);
  std::printf("\n");
}

TEST_F(Gpt2Forward, RejectsTooManyTokens) {
  ForwardInput in;
  in.n_tokens = 10000;
  EXPECT_THROW(model_->forward(in, *kv_, d_logits_, nullptr), std::invalid_argument);
}
