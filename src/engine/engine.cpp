#include "engine/engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>

#include <cublas_v2.h>

#include "core/cuda_check.h"
#include "core/logging.h"
#include "core/memory.h"
#include "engine/cuda_graph.h"
#include "kernels/kernels.h"
#include "kv_cache/kv_cache.h"
#include "model/config.h"
#include "model/registry.h"
#include "model/safetensors.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/tokenizer.h"

namespace ography {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr size_t kMiB = size_t{1} << 20;

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string mib(size_t bytes) { return std::to_string(bytes / kMiB) + " MiB"; }

// Weights are stored fp16 on the GPU whatever the checkpoint dtype.
// TODO: account for quantized weights (roadmap step 9).
size_t estimate_weight_bytes(const std::string& model_dir) {
  SafetensorsModel ckpt(model_dir);
  size_t bytes = 0;
  for (const auto& name : ckpt.names()) bytes += ckpt.get(name).numel() * 2;
  return bytes;
}

// Generous upper bound on per-forward-pass activations for `tokens` tokens, plus the engine's
// own per-vocab buffers.
size_t estimate_scratch_bytes(const ModelConfig& c, int tokens) {
  const size_t per_token = 16 * size_t(c.hidden_size) + 4 * size_t(c.intermediate_size);
  return size_t(tokens) * per_token + 16 * size_t(c.vocab_size) + 16 * kMiB;
}

}  // namespace

struct Engine::Impl {
  EngineConfig config;
  std::string name;
  cudaStream_t stream = nullptr;
  cublasHandle_t cublas = nullptr;
  std::unique_ptr<DeviceArena> arena;
  std::unique_ptr<Model> model;
  std::unique_ptr<KVCache> kv;
  std::unique_ptr<Tokenizer> tokenizer;
  ChatTemplate chat_template;
  int chunk = 0;           // tokens per prefill pass
  int sample_vocab = 0;    // min(model vocab, tokenizer vocab): never sample padding rows

  // Fixed device buffers. d_ids[0] doubles as the sampled token, so decode reads it in place.
  Tensor d_ids, d_pos, d_logits, d_probs, d_prompt_mask, d_output_counts;
  int32_t* h_pinned = nullptr;  // [0] sampled token, [1] decode position

  ~Impl() {
    model.reset();
    if (h_pinned) cudaFreeHost(h_pinned);
    if (cublas) cublasDestroy(cublas);
    if (stream) cudaStreamDestroy(stream);
  }
};

Engine::Engine(const EngineConfig& config) : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  m.config = config;
  const fs::path dir = config.model_dir;
  m.name = fs::path(config.model_dir).filename().string();
  if (m.name.empty()) m.name = dir.parent_path().filename().string();

  const ModelConfig mc = ModelConfig::from_file((dir / "config.json").string());
  m.tokenizer = std::make_unique<Tokenizer>(Tokenizer::from_file((dir / "tokenizer.json").string()));
  m.chat_template = ChatTemplate::from_tokenizer_config((dir / "tokenizer_config.json").string());
  m.sample_vocab = std::min(mc.vocab_size, m.tokenizer->vocab_size());

  // Memory plan: [weights | scratch + engine buffers | KV cache], one allocation.
  int want_ctx = mc.max_position_embeddings;
  if (config.max_seq_len > 0) want_ctx = std::min(want_ctx, config.max_seq_len);
  m.chunk = std::max(1, std::min(config.prefill_chunk, want_ctx));
  KVCacheConfig kvc{mc.num_layers, mc.num_kv_heads, mc.head_dim, 0};
  const size_t kv_per_token = KVCache::bytes_per_token(kvc);
  const size_t fixed = estimate_weight_bytes(config.model_dir) +
                       estimate_scratch_bytes(mc, m.chunk) + 64 * kMiB;
  const size_t budget = static_cast<size_t>(config.vram_fraction * query_free_device_memory());
  const size_t min_ctx = std::min(want_ctx, 256);
  if (budget < fixed + min_ctx * kv_per_token) {
    throw std::runtime_error("not enough free VRAM: need about " +
                             mib(fixed + min_ctx * kv_per_token) + ", have " + mib(budget));
  }
  m.arena = std::make_unique<DeviceArena>(std::min(budget, fixed + want_ctx * kv_per_token));

  CUDA_CHECK(cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasCreate(&m.cublas));
  CUBLAS_CHECK(cublasSetStream(m.cublas, m.stream));

  const auto t0 = Clock::now();
  m.model = load_model(config.model_dir, *m.arena, m.cublas, m.chunk);
  const size_t weights_and_scratch = m.arena->used();

  const int64_t vocab = mc.vocab_size;
  m.d_ids = m.arena->allocate(DType::I32, {m.chunk});
  m.d_pos = m.arena->allocate(DType::I32, {m.chunk});
  m.d_logits = m.arena->allocate(DType::F32, {vocab});
  m.d_probs = m.arena->allocate(DType::F32, {vocab});
  m.d_prompt_mask = m.arena->allocate(DType::I8, {vocab});
  m.d_output_counts = m.arena->allocate(DType::I32, {vocab});
  CUDA_CHECK(cudaMallocHost(&m.h_pinned, 2 * sizeof(int32_t)));

  // Whatever is left (less alignment slack) goes to the KV cache.
  const size_t left = m.arena->capacity() - m.arena->used() - kMiB;
  kvc.max_seq_len = static_cast<int>(std::min<size_t>(want_ctx, left / kv_per_token));
  m.kv = std::make_unique<KVCache>(kvc, *m.arena);

  // Warm-up: the first kernel launches and cuBLAS calls pay one-off setup costs (module
  // loading, heuristics), which would otherwise land on the first request's TTFT.
  const auto t_warm = Clock::now();
  for (int n : {std::min(m.chunk, 16), 1}) {
    CUDA_CHECK(cudaMemsetAsync(m.d_ids.data(), 0, n * 4, m.stream));
    CUDA_CHECK(cudaMemsetAsync(m.d_pos.data(), 0, n * 4, m.stream));
    ForwardInput in;
    in.token_ids = m.d_ids.slice_rows(0, n);
    in.positions = m.d_pos.slice_rows(0, n);
    in.n_tokens = n;
    in.is_decode = n == 1;
    m.model->forward(in, *m.kv, m.d_logits, m.stream);
  }
  kernels::SamplingKernelParams warm;
  kernels::sample(m.d_logits.slice_rows(0, m.sample_vocab), warm, m.d_ids.data_as<int32_t>(),
                  m.d_probs, m.stream);
  CUDA_CHECK(cudaStreamSynchronize(m.stream));
  const double warm_ms = ms_between(t_warm, Clock::now());

  char msg[256];
  std::snprintf(msg, sizeof msg,
                "loaded %s (%s) in %.0f ms (warm-up %.0f ms): arena %s, weights+scratch %s, "
                "KV cache %d tokens (%s)",
                m.name.c_str(), mc.architecture.c_str(), ms_between(t0, Clock::now()), warm_ms,
                mib(m.arena->capacity()).c_str(), mib(weights_and_scratch).c_str(),
                kvc.max_seq_len, mib(kvc.max_seq_len * kv_per_token).c_str());
  log(LogLevel::Info, msg);
}

Engine::~Engine() = default;

const Tokenizer& Engine::tokenizer() const { return *impl_->tokenizer; }
const ChatTemplate& Engine::chat_template() const { return impl_->chat_template; }
const std::string& Engine::model_name() const { return impl_->name; }
int Engine::max_seq_len() const { return impl_->kv->capacity(); }

GenerationResult Engine::generate(const GenerationRequest& request,
                                  const TokenCallback& on_token) {
  Impl& m = *impl_;
  const auto& prompt = request.prompt_tokens;
  const int n_prompt = static_cast<int>(prompt.size());
  if (n_prompt == 0) throw std::invalid_argument("generate: empty prompt");
  if (n_prompt >= m.kv->capacity()) {
    throw std::invalid_argument("generate: prompt of " + std::to_string(n_prompt) +
                                " tokens doesn't fit the " + std::to_string(m.kv->capacity()) +
                                "-token context");
  }
  for (int32_t id : prompt) {
    if (id < 0 || id >= m.sample_vocab) throw std::invalid_argument("generate: bad token id");
  }

  // The last sampled token is never fed back, so g tokens need n_prompt + g - 1 cache slots.
  SamplingParams sp = request.sampling;
  sp.max_tokens = std::max(1, std::min(sp.max_tokens, m.kv->capacity() - n_prompt + 1));
  StopChecker stop(sp, m.tokenizer->eos_ids());
  IncrementalDecoder decoder(*m.tokenizer);

  kernels::SamplingKernelParams kp;
  kp.temperature = sp.temperature;
  kp.top_k = sp.top_k;
  kp.top_p = sp.top_p;
  kp.min_p = sp.min_p;
  kp.seed = sp.seed != 0 ? sp.seed : (uint64_t(std::random_device{}()) << 32 | std::random_device{}());
  const bool penalties = sp.repetition_penalty != 1.f || sp.presence_penalty != 0.f ||
                         sp.frequency_penalty != 0.f;
  Tensor logits = m.d_logits.slice_rows(0, m.sample_vocab);

  GenerationResult result;
  result.prompt_tokens = n_prompt;
  const auto t_start = Clock::now();
  Clock::time_point t_first;

  // TODO(step 8): reuse the cached prefix from the previous turn instead of clearing.
  m.kv->clear();
  if (penalties) {
    CUDA_CHECK(cudaMemsetAsync(m.d_prompt_mask.data(), 0, m.d_prompt_mask.nbytes(), m.stream));
    CUDA_CHECK(cudaMemsetAsync(m.d_output_counts.data(), 0, m.d_output_counts.nbytes(), m.stream));
  }

  // Prefill in chunks. Async copies from pageable memory are staged before returning, and the
  // stream orders each chunk's upload after the previous chunk's forward pass.
  std::vector<int32_t> pos(m.chunk);
  for (int off = 0; off < n_prompt; off += m.chunk) {
    const int n = std::min(m.chunk, n_prompt - off);
    for (int i = 0; i < n; ++i) pos[i] = off + i;
    CUDA_CHECK(cudaMemcpyAsync(m.d_ids.data(), prompt.data() + off, n * 4,
                               cudaMemcpyHostToDevice, m.stream));
    CUDA_CHECK(cudaMemcpyAsync(m.d_pos.data(), pos.data(), n * 4, cudaMemcpyHostToDevice,
                               m.stream));
    ForwardInput in;
    in.token_ids = m.d_ids.slice_rows(0, n);
    in.positions = m.d_pos.slice_rows(0, n);
    in.n_tokens = n;
    in.start_pos = off;
    in.is_decode = false;
    if (penalties) kernels::mark_tokens(in.token_ids, m.d_prompt_mask, m.stream);
    m.model->forward(in, *m.kv, m.d_logits, m.stream);
    m.kv->advance(n);
  }

  int32_t* d_token = m.d_ids.data_as<int32_t>();
  for (int generated = 1;; ++generated) {
    if (penalties) {
      kernels::apply_penalties(logits, m.d_prompt_mask, m.d_output_counts, sp.repetition_penalty,
                               sp.presence_penalty, sp.frequency_penalty, m.stream);
    }
    kp.step = static_cast<uint64_t>(generated);
    kernels::sample(logits, kp, d_token, m.d_probs, m.stream);
    if (penalties) kernels::count_token(d_token, m.d_output_counts, m.stream);
    CUDA_CHECK(cudaMemcpyAsync(&m.h_pinned[0], d_token, 4, cudaMemcpyDeviceToHost, m.stream));
    CUDA_CHECK(cudaStreamSynchronize(m.stream));
    const int32_t token = m.h_pinned[0];
    if (generated == 1) t_first = Clock::now();
    result.completion_tokens = generated;

    // Stop tokens' own text is never shown; at the limit, flush any partial character.
    std::string piece = stop.is_stop_token(token) ? decoder.flush() : decoder.push(token);
    if (generated >= sp.max_tokens) piece += decoder.flush();
    StopChecker::Result r = stop.push(token, piece, generated);

    if (!on_token(TokenEvent{token, std::move(r.emit)})) {
      result.finish_reason = FinishReason::Cancelled;
      break;
    }
    if (r.finish) {
      result.finish_reason = *r.finish;
      break;
    }

    // Feed the sampled token back: it is already in d_ids[0].
    m.h_pinned[1] = m.kv->seq_len();
    CUDA_CHECK(cudaMemcpyAsync(m.d_pos.data(), &m.h_pinned[1], 4, cudaMemcpyHostToDevice,
                               m.stream));
    ForwardInput in;
    in.token_ids = m.d_ids.slice_rows(0, 1);
    in.positions = m.d_pos.slice_rows(0, 1);
    in.n_tokens = 1;
    in.start_pos = m.kv->seq_len();
    in.is_decode = true;
    m.model->forward(in, *m.kv, m.d_logits, m.stream);
    m.kv->advance(1);
  }

  const auto t_end = Clock::now();
  result.time_to_first_token_ms = ms_between(t_start, t_first);
  if (result.completion_tokens > 1) {
    result.tokens_per_second = (result.completion_tokens - 1) / (ms_between(t_first, t_end) / 1e3);
  }
  return result;
}

}  // namespace ography
