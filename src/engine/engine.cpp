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
#include "kv_cache/prefix_reuse.h"
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
// A fixed cuBLAS workspace, so cuBLAS never allocates (which CUDA-graph capture forbids).
constexpr size_t kCublasWorkspace = 32 * kMiB;

// Host-side state the decode graph reads/writes at replay time; must be pinned memory.
struct PinnedState {
  int32_t token;     // sampled token (device -> host each step)
  int32_t position;  // position of the token being fed (host -> device each step)
  kernels::SamplingKernelParams params;  // uploaded once per request
};

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string mib(size_t bytes) { return std::to_string(bytes / kMiB) + " MiB"; }

// Float weights are stored fp16 on the GPU whatever the checkpoint dtype; packed INT4 weights
// keep their size, and their zero points grow from 4 to 16 bits.
size_t estimate_weight_bytes(const std::string& model_dir, const ModelConfig& config) {
  SafetensorsModel ckpt(model_dir);
  size_t bytes = 0;
  for (const auto& name : ckpt.names()) {
    // Some tied checkpoints still ship lm_head.weight; the engine uses the embeddings.
    if (config.tie_word_embeddings && name == "lm_head.weight") continue;
    const Tensor t = ckpt.get(name);
    const bool is_float = t.dtype() == DType::F32 || t.dtype() == DType::F16 || t.dtype() == DType::BF16;
    if (name.size() >= 6 && name.compare(name.size() - 6, 6, ".g_idx") == 0) continue;  // not kept
    const bool qzeros = name.size() >= 7 && name.compare(name.size() - 7, 7, ".qzeros") == 0;
    bytes += is_float ? t.numel() * 2 : qzeros ? t.nbytes() * 4 : t.nbytes();
  }
  return bytes;
}

// Generous upper bound on per-forward-pass activations for `tokens` tokens, plus the engine's
// own per-vocab buffers.
size_t estimate_scratch_bytes(const ModelConfig& c, int tokens) {
  const size_t per_token = 16 * size_t(c.hidden_size) + 4 * size_t(c.intermediate_size);
  const size_t dequant = c.quant.method != QuantMethod::None ? 8 * kMiB : 0;  // see LlamaModel
  return size_t(tokens) * per_token + 16 * size_t(c.vocab_size) + 16 * kMiB + kCublasWorkspace +
         dequant;
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
  Tensor d_ids, d_pos, d_logits, d_probs, d_prompt_mask, d_output_counts, d_params_buf;
  kernels::SamplingKernelParams* d_params = nullptr;  // in d_params_buf
  PinnedState* h = nullptr;

  PrefixTracker prefix;  // what the KV cache holds, for reuse by the next request
  DecodeGraph graph;

  Tensor sample_logits() const { return d_logits.slice_rows(0, sample_vocab); }

  // Penalties, sampling (into d_ids[0]), output counts, copy of the token to the host.
  void sample_step(cudaStream_t s) {
    Tensor logits = sample_logits();
    kernels::apply_penalties(logits, d_prompt_mask, d_output_counts, d_params, s);
    kernels::sample(logits, d_params, d_ids.data_as<int32_t>(), d_probs, s);
    kernels::count_token(d_ids.data_as<int32_t>(), d_output_counts, s);
    CUDA_CHECK(cudaMemcpyAsync(&h->token, d_ids.data(), 4, cudaMemcpyDeviceToHost, s));
  }

  // One decode step: feed d_ids[0] at position h->position, then sample the next token.
  // Everything per-step is read from device or pinned memory, so this is graph-capturable.
  void decode_step(cudaStream_t s) {
    CUDA_CHECK(cudaMemcpyAsync(d_pos.data(), &h->position, 4, cudaMemcpyHostToDevice, s));
    ForwardInput in;
    in.token_ids = d_ids.slice_rows(0, 1);
    in.positions = d_pos.slice_rows(0, 1);
    in.n_tokens = 1;
    in.start_pos = kv->seq_len();  // host-side validation only
    in.is_decode = true;
    model->forward(in, *kv, d_logits, s);
    sample_step(s);
  }

  ~Impl() {
    graph.reset();
    model.reset();
    if (h) cudaFreeHost(h);
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
  KVCacheConfig kvc{mc.num_layers, mc.num_kv_heads, mc.head_dim, 0,
                    config.fp8_kv_cache ? DType::FP8_E4M3 : DType::F16, mc.num_heads};
  const size_t kv_per_token = KVCache::bytes_per_token(kvc);
  if (config.fp8_kv_cache && mc.attention_bias && mc.model_type != "gpt2") {
    log(LogLevel::Warn,
        "--fp8-kv loses noticeable accuracy on models with q/k/v biases (Qwen2): their K "
        "values have large outlier channels that E4M3 can't represent precisely");
  }
  KVCacheConfig full = kvc;
  full.max_seq_len = want_ctx;
  const size_t fixed = KVCache::scratch_bytes(full) + estimate_weight_bytes(config.model_dir, mc) +
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
  // Prefer bit-reproducible kernels so graph replay and eager runs give identical results.
  CUBLAS_CHECK(cublasSetAtomicsMode(m.cublas, CUBLAS_ATOMICS_NOT_ALLOWED));

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
  m.d_params_buf = m.arena->allocate(DType::I8, {int64_t(sizeof(kernels::SamplingKernelParams))});
  m.d_params = static_cast<kernels::SamplingKernelParams*>(m.d_params_buf.data());
  Tensor workspace = m.arena->allocate(DType::I8, {int64_t(kCublasWorkspace)});
  CUBLAS_CHECK(cublasSetWorkspace(m.cublas, workspace.data(), kCublasWorkspace));
  CUDA_CHECK(cudaMallocHost(&m.h, sizeof(PinnedState)));
  *m.h = PinnedState{};
  CUDA_CHECK(cudaMemsetAsync(m.d_prompt_mask.data(), 0, m.d_prompt_mask.nbytes(), m.stream));
  CUDA_CHECK(cudaMemsetAsync(m.d_output_counts.data(), 0, m.d_output_counts.nbytes(), m.stream));

  // Whatever is left (less alignment slack) goes to the KV cache.
  // Decode scratch also grows with capacity; it is at most ~1% of the cache, so reserve its
  // size for the full-context case and size the cache from the rest.
  KVCacheConfig probe = kvc;
  probe.max_seq_len = want_ctx;
  const size_t left = m.arena->capacity() - m.arena->used() - kMiB - KVCache::scratch_bytes(probe);
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
  CUDA_CHECK(cudaMemcpyAsync(m.d_params, &m.h->params, sizeof(m.h->params),
                             cudaMemcpyHostToDevice, m.stream));
  m.sample_step(m.stream);
  CUDA_CHECK(cudaStreamSynchronize(m.stream));
  // Capture the decode step once; every later step of every request replays it.
  if (config.use_cuda_graphs) m.graph.capture(m.stream, [&m](cudaStream_t s) { m.decode_step(s); });
  const double warm_ms = ms_between(t_warm, Clock::now());

  char msg[256];
  std::snprintf(msg, sizeof msg,
                "loaded %s (%s) in %.0f ms (warm-up %.0f ms%s): arena %s, weights+scratch %s, "
                "KV cache %d tokens (%s, %s)",
                m.name.c_str(), mc.architecture.c_str(), ms_between(t0, Clock::now()), warm_ms,
                config.use_cuda_graphs ? ", decode graph captured" : "",
                mib(m.arena->capacity()).c_str(), mib(weights_and_scratch).c_str(),
                kvc.max_seq_len, mib(kvc.max_seq_len * kv_per_token).c_str(),
                std::string(dtype_name(kvc.dtype)).c_str());
  log(LogLevel::Info, msg);
}

Engine::~Engine() = default;

const Tokenizer& Engine::tokenizer() const { return *impl_->tokenizer; }
const ChatTemplate& Engine::chat_template() const { return impl_->chat_template; }
const std::string& Engine::model_name() const { return impl_->name; }
int Engine::max_seq_len() const { return impl_->kv->capacity(); }

void Engine::clear_cache() {
  impl_->kv->clear();
  impl_->prefix.clear();
}

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

  kernels::SamplingKernelParams& kp = m.h->params;
  kp = kernels::SamplingKernelParams{};
  kp.temperature = sp.temperature;
  kp.top_k = sp.top_k;
  kp.top_p = sp.top_p;
  kp.min_p = sp.min_p;
  kp.seed = sp.seed != 0 ? sp.seed : (uint64_t(std::random_device{}()) << 32 | std::random_device{}());
  kp.step = 1;
  kp.repetition = sp.repetition_penalty;
  kp.presence = sp.presence_penalty;
  kp.frequency = sp.frequency_penalty;
  const bool penalties = kp.repetition != 1.f || kp.presence != 0.f || kp.frequency != 0.f;

  GenerationResult result;
  result.prompt_tokens = n_prompt;
  const auto t_start = Clock::now();
  Clock::time_point t_first;

  // Reuse whatever prefix of this prompt the cache already holds (in a chat, usually the
  // whole previous conversation), and prefill only the rest.
  const int reuse = static_cast<int>(m.prefix.reusable_prefix(prompt));
  result.cached_prompt_tokens = reuse;
  m.kv->truncate(std::min(reuse, m.kv->seq_len()));
  std::vector<int32_t> cached(prompt.begin(), prompt.begin() + m.kv->seq_len());
  m.prefix.clear();  // until this request completes, the cache contents are in flux

  try {
    CUDA_CHECK(cudaMemcpyAsync(m.d_params, &kp, sizeof(kp), cudaMemcpyHostToDevice, m.stream));
    if (penalties) {
      CUDA_CHECK(cudaMemsetAsync(m.d_prompt_mask.data(), 0, m.d_prompt_mask.nbytes(), m.stream));
      CUDA_CHECK(cudaMemsetAsync(m.d_output_counts.data(), 0, m.d_output_counts.nbytes(), m.stream));
    }

    // Prefill in chunks. Async copies from pageable memory are staged before returning, and
    // the stream orders each chunk's upload after the previous chunk's forward pass. With
    // penalties on, the reused prefix is uploaded too, just to mark it in the prompt mask.
    std::vector<int32_t> pos(m.chunk);
    for (int off = penalties ? 0 : m.kv->seq_len(); off < n_prompt; off += m.chunk) {
      const int n = std::min(m.chunk, n_prompt - off);
      ForwardInput in;
      in.token_ids = m.d_ids.slice_rows(0, n);
      CUDA_CHECK(cudaMemcpyAsync(m.d_ids.data(), prompt.data() + off, n * 4,
                                 cudaMemcpyHostToDevice, m.stream));
      if (penalties) kernels::mark_tokens(in.token_ids, m.d_prompt_mask, m.stream);
      // Forward only the part not already cached (a chunk can straddle the boundary).
      const int skip = std::max(0, m.kv->seq_len() - off);
      if (skip >= n) continue;
      if (skip > 0) {
        CUDA_CHECK(cudaMemcpyAsync(m.d_ids.data(), prompt.data() + off + skip, (n - skip) * 4,
                                   cudaMemcpyHostToDevice, m.stream));
      }
      const int fwd = n - skip;
      for (int i = 0; i < fwd; ++i) pos[i] = off + skip + i;
      CUDA_CHECK(cudaMemcpyAsync(m.d_pos.data(), pos.data(), fwd * 4, cudaMemcpyHostToDevice,
                                 m.stream));
      in.token_ids = m.d_ids.slice_rows(0, fwd);
      in.positions = m.d_pos.slice_rows(0, fwd);
      in.n_tokens = fwd;
      in.start_pos = off + skip;
      in.is_decode = false;
      m.model->forward(in, *m.kv, m.d_logits, m.stream);
      m.kv->advance(fwd);
      cached.insert(cached.end(), prompt.begin() + off + skip, prompt.begin() + off + n);
    }

    m.sample_step(m.stream);
    for (int generated = 1;; ++generated) {
      CUDA_CHECK(cudaStreamSynchronize(m.stream));
      const int32_t token = m.h->token;
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

      // Feed the sampled token (already in d_ids[0]) back and sample the next one.
      m.h->position = m.kv->seq_len();
      if (m.graph.captured()) {
        m.graph.replay(m.stream);
      } else {
        m.decode_step(m.stream);
      }
      m.kv->advance(1);
      cached.push_back(token);
    }
  } catch (...) {
    // The cache may be half-written; start the next request from scratch.
    m.kv->clear();
    m.prefix.clear();
    throw;
  }
  m.prefix.set_cached(std::move(cached));

  const auto t_end = Clock::now();
  result.time_to_first_token_ms = ms_between(t_start, t_first);
  if (result.completion_tokens > 1) {
    result.tokens_per_second = (result.completion_tokens - 1) / (ms_between(t_first, t_end) / 1e3);
  }
  return result;
}

}  // namespace ography
