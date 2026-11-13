# Ography

A from-scratch LLM inference engine written in C++ and CUDA, built for **low-latency,
single-user inference** on a consumer laptop GPU. It loads Hugging Face checkpoints
(`config.json` + `*.safetensors` + `tokenizer.json`), runs the forward pass on custom CUDA
kernels, and serves the model through an **OpenAI-compatible HTTP API** so existing
OpenAI clients and tools work against it unchanged.

> **Status: early.** GPT-2 and the Llama family (Llama 3.x, Qwen2/2.5, Mistral), in fp16 or
> INT4 (AWQ / GPTQ), run end to end, matching HF `transformers` layer by layer and token for
> token, with the tokenizer and chat templates matching HF exactly. A 7B INT4 model runs in
> 8 GB. They are served through an OpenAI-compatible API
> (`ography_server`) or a terminal chat (`ography_cli`), reusing the KV cache across chat
> turns. See the [roadmap](#roadmap) for progress.

---

## Contents

- [Goals](#goals)
- [Target hardware](#target-hardware)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Building](#building)
- [Running](#running)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Memory planning on 8 GB](#memory-planning-on-8-gb)
- [Roadmap](#roadmap)
- [Conventions](#conventions)

---

## Goals

**Goals**
- Lowest possible latency for **one client**: time-to-first-token and time per output token.
- Run off-the-shelf Hugging Face decoder models. GPT-2 (`openai-community/gpt2`) is the
  bring-up model; the Llama family (Llama, Mistral, Qwen2) follows as the daily-use target.
- Own the full stack: weight loading, kernels, tokenizer, sampling, and HTTP server.
- Be numerically verified against HF `transformers`, layer by layer.

**Non-goals**
- Multi-user serving. There is no continuous batching, paged KV cache or request
  scheduling; the GPU serves one request at a time.
- Training or fine-tuning.
- Python at runtime. Python is used only for offline tooling and golden reference data.

## Target hardware

| | |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop (Ada Lovelace) |
| Compute capability | `sm_89` (native FP8 tensor cores) |
| VRAM | 8 GB |
| Memory bandwidth | ~256 GB/s |
| Toolchain | CUDA ≥ 12.4 (tested with 13.5), GCC 13, CMake ≥ 3.24 |

At batch size 1, decoding is **memory-bandwidth bound**: every generated token reads every
weight once. Token rate is therefore capped near `bandwidth / model bytes`. For example, an
8B model with ~4.5 GB of INT4 weights has a ceiling of about 55 tokens/s on this card.
Quantization and fewer kernel launches are the biggest levers, and the design is built
around them.

## How it works

```
 HTTP client (OpenAI SDK, curl, any OpenAI-compatible tool)
      │  POST /v1/chat/completions
      ▼
 ┌──────────────┐   messages ─► ChatTemplate ─► Tokenizer ─► token ids
 │  server/     │
 │  HttpServer  │──── job ────► RequestQueue (single worker thread owns the GPU)
 └──────────────┘                     │
      ▲  SSE chunks                   ▼
      │                        ┌──────────────┐
      └──── TokenCallback ◄────│  Engine      │
                               │  generate()  │
                               └──────┬───────┘
          ┌───────────────────────────┼─────────────────────────────┐
          ▼                           ▼                             ▼
   PrefixTracker               Model::forward                  sampling kernel
   (reuse KV from the          (Gpt2Model / LlamaModel:        (penalties, temperature,
    previous chat turn)          N × decoder layer →            top-k/p, min-p; result
          │                      final norm → lm_head)          stays on the GPU)
          ▼                           │
       KVCache ◄──────────────────────┘
```

Each request goes through these steps:

1. **Parse.** The server turns the OpenAI request into chat messages and `SamplingParams`.
2. **Tokenize.** The model's chat template is rendered and encoded into token ids.
3. **Reuse the prefix.** In a multi-turn chat the new prompt begins with the whole previous
   conversation, which is already in the KV cache. Only the new suffix is processed.
4. **Prefill.** The remaining prompt tokens run through the model in chunks. This step is
   compute-bound, uses cuBLAS GEMMs and tiled attention, and fills the KV cache.
5. **Decode.** One token per step. The forward pass and sampling are captured once as a
   **CUDA graph** and replayed, removing per-kernel launch overhead. Only the sampled
   token id (4 bytes) is copied back to the host each step.
6. **Stream.** Token ids are decoded to UTF-8-safe text, checked against stop tokens and
   stop strings, and streamed to the client as server-sent events. If the client
   disconnects, generation is cancelled.

## Repository layout

```
ography/
├── CMakeLists.txt              # top-level build (C++20, CUDA sm_89)
├── cmake/
│   ├── CompilerFlags.cmake     # warning / optimisation flags (ography_set_flags)
│   └── Dependencies.cmake      # nlohmann/json, cpp-httplib, GoogleTest via FetchContent
├── src/                        # one static library per module: ography_<module>
│   ├── core/                   # Tensor (non-owning view), DType, DeviceArena, CUDA_CHECK, logging
│   ├── model/                  # config.json, mmapped safetensors, HF→engine weight mapping
│   │   └── arch/               # model architectures (gpt2, llama: also Qwen2 / Mistral)
│   ├── kernels/                # one .cu per kernel, all declared in kernels.h
│   ├── kv_cache/               # contiguous per-layer KV cache, prefix reuse tracker
│   ├── tokenizer/              # byte-level BPE, chat templates, incremental UTF-8 decoding
│   ├── quant/                  # quantization configs (FP8, AWQ, GPTQ) and weight structs
│   ├── engine/                 # generate loop, sampling params, stop checker, CUDA graphs, queue
│   └── server/                 # HTTP server, OpenAI JSON schema, SSE framing (target: ography_http)
├── apps/
│   ├── ography_server.cpp      # OpenAI-compatible server binary
│   ├── ography_cli.cpp         # terminal chat, no HTTP (fastest debugging loop)
│   └── ography_bench.cpp       # end-to-end latency benchmark
├── tests/
│   ├── unit/                   # pure C++ logic (GoogleTest)
│   ├── kernels/                # each kernel vs a CPU reference
│   ├── golden/                 # engine vs HF transformers, layer by layer
│   └── data/tokenizer/         # HF encodings + chat renderings per tokenizer (checked in)
├── bench/                      # kernel microbenchmarks (cudaEvent timing)
├── tools/                      # Python, offline only
│   ├── dump_golden.py          # dump HF reference tensors for tests/golden
│   ├── dump_tokenizer_cases.py # dump HF tokenizer / chat-template fixtures for tests/data
│   ├── gen_unicode_tables.py   # regenerate src/tokenizer/unicode_tables.inc
│   ├── inspect_safetensors.py  # list tensor names / dtypes / shapes in a checkpoint
│   └── requirements.txt
├── third_party/                # reserved for vendored deps (none yet)
└── docs/
    └── architecture.md         # design constraints, request flow, memory plan
```

### Module dependencies

Dependencies flow one way only. Lower layers never include higher ones.

```
core ◄── kernels ◄── kv_cache ◄── model ◄── engine ◄── server
core ◄── quant   ◄── model
tokenizer (CPU only) ◄── engine
```

| Module | Key types | Responsibility |
|---|---|---|
| `core` | `Tensor`, `DType`, `DeviceArena` | Memory views, one-shot GPU allocation, error handling |
| `kernels` | `kernels::layernorm`, `split_qkv`, `attention_prefill`, `gemm`, … | All GPU compute. Async, stream-ordered, CUDA-graph safe |
| `kv_cache` | `KVCache`, `PrefixTracker` | Per-layer K/V storage; reusing cached prompt prefixes |
| `quant` | `QuantConfig`, `QuantizedWeight` | Describes quantized checkpoint formats |
| `model` | `ModelConfig`, `SafetensorsModel`, `Gpt2Weights` / `LlamaWeights`, `Model` | Loading, mapping and running a transformer |
| `tokenizer` | `Tokenizer`, `IncrementalDecoder`, `ChatTemplate` | Text ↔ token ids (byte-level BPE, hand-written pretokenizers), prompt formatting |
| `engine` | `Engine`, `SamplingParams`, `StopChecker`, `DecodeGraph`, `RequestQueue` | The prefill/decode loop and its control flow |
| `server` | `HttpServer`, `openai::*`, `sse::*` | OpenAI-compatible HTTP surface |

## Prerequisites

### 1. NVIDIA driver

Your driver must support the CUDA version you install. Check with `nvidia-smi`: the
"CUDA Version" it reports is the maximum supported.

### 2. CUDA toolkit (Ubuntu 24.04 / Pop!_OS)

Ubuntu's own `nvidia-cuda-toolkit` package is CUDA 12.0, which does not support GCC 13.
Install from NVIDIA's repository instead:

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-2 ninja-build
```

Then add it to your shell (`~/.bashrc`):

```sh
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Verify with `nvcc --version`.

The toolkit may be newer than the CUDA version `nvidia-smi` reports (e.g. toolkit 13.5 on a
13.2 driver). That works within the same major version because the build compiles native
`sm_89` code, so no PTX JIT by the older driver is needed.

### 3. Build tools

- CMake ≥ 3.24 and GCC 13 (`sudo apt install cmake g++`)
- Ninja (optional, faster builds)

C++ dependencies are downloaded automatically by CMake at configure time:

| Library | Version | Used for |
|---|---|---|
| [nlohmann/json](https://github.com/nlohmann/json) | 3.11.3 | config, tokenizer and API JSON |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | 0.18.1 | HTTP server |
| [GoogleTest](https://github.com/google/googletest) | 1.15.2 | tests |
| cuBLAS | (CUDA toolkit) | matrix multiplies, until custom GEMM kernels replace it |

### 4. Python tooling (optional, for golden tests)

Used only offline, to produce reference tensors from HF `transformers`. CPU-only PyTorch is
enough: references are computed in fp32.

```sh
uv venv tools/.venv --python 3.12
VIRTUAL_ENV=tools/.venv uv pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  -r tools/requirements.txt
```

(Plain `python -m venv` + `pip install -r tools/requirements.txt` also works.)

### 5. A model

Download any Hugging Face checkpoint in safetensors format. Llama weights are gated: accept
the license on the model page and log in first (`huggingface-cli login`).

```sh
tools/.venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/qwen2.5-1.5b
tools/.venv/bin/hf download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir models/qwen2.5-7b-awq
tools/.venv/bin/python tools/inspect_safetensors.py models/qwen2.5-1.5b
```

Supported: GPT-2, and Llama / Mistral / Qwen2 checkpoints in fp16 / bf16 / fp32, or INT4
AWQ ("gemm" packing) and GPTQ (without act-order). 4-bit 7–8B models fit in 8 GB.

`models/` is git-ignored.

## Building

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

| CMake option | Default | Meaning |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug` for `-O0 -g`; CUDA always keeps `-lineinfo` for profiling |
| `CMAKE_CUDA_ARCHITECTURES` | `89` | Override for other GPUs (e.g. `86` for Ampere, `90` for Hopper) |
| `OGRAPHY_BUILD_TESTS` | `ON` | Build `unit_tests`, `kernel_tests`, `golden_tests` |
| `OGRAPHY_BUILD_BENCH` | `ON` | Build `bench_kernels` |

If CMake can't find `nvcc`, pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.

`compile_commands.json` is generated in `build/` for clangd and other IDE tooling.

## Running

### Server

```sh
./build/apps/ography_server --model models/qwen2.5-1.5b --port 8000
```

| Flag | Meaning |
|---|---|
| `--model DIR` | Hugging Face model directory |
| `--host`, `--port` | Bind address (defaults to `127.0.0.1:8000`, so it's local only) |
| `--max-seq-len N` | Context length; default is as large as free VRAM allows |
| `--prefill-chunk N` | Longest prompt slice per forward pass (default 2048) |
| `--vram-fraction F` | Share of free VRAM to claim (default 0.92) |
| `--no-cuda-graphs` | Decode without graph replay (debugging) |
| `--fp8-kv` | Store the KV cache in FP8: twice the context (not recommended for Qwen2) |
| `--api-key KEY` | Require `Authorization: Bearer KEY` on `/v1/*` (or set `OGRAPHY_API_KEY`) |

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions` (both streaming and not),
`GET /v1/models`, `GET /health` (status plus request / in-flight / cancelled counters).

Requests run one at a time on the GPU, in arrival order. Supported request fields: `messages`
(text content), `prompt`, `max_tokens` / `max_completion_tokens` (default: as much as fits),
`temperature`, `top_p`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `stream`,
`stream_options.include_usage`, plus vLLM-style `top_k`, `min_p`, `repetition_penalty`.
Anything the engine can't honour (`n > 1`, tools, logprobs, images, JSON mode) is rejected
with a 400 rather than ignored. Closing a streaming connection cancels generation. `seed`
reproduces outputs for the same request from the same cache state; with a warm prefix cache
the rounding can differ slightly, so like OpenAI's it is best effort.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "local", "stream": true,
       "messages": [{"role": "user", "content": "Explain KV caching in one paragraph."}]}'
```

With the official OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
for chunk in client.chat.completions.create(
        model="local", stream=True,
        messages=[{"role": "user", "content": "Hello!"}]):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The server listens on localhost only. To reach it from another machine, put a TLS reverse
proxy (e.g. Caddy) in front and set `--api-key`.

### Terminal chat

```sh
./build/apps/ography_cli --model models/gpt2                 # completion mode (no chat template)
./build/apps/ography_cli --model models/qwen2.5-1.5b --temperature 0.7   # chat
```

Models with a chat template get a multi-turn chat (`/reset` clears it); models without one,
like GPT-2, get completion mode, where each line is a prompt to continue. Ctrl-C stops a
reply. Options: `--temperature`, `--greedy`, `--top-k`, `--top-p`, `--min-p`,
`--repetition-penalty`, `--max-tokens`, `--seed`, `--system`, `--raw`, `--max-seq-len`,
`--prefill-chunk` (`--help` for defaults). After each reply it prints prompt+completion
tokens, time to first token and decode tokens/s.

## Testing

```sh
ctest --test-dir build --output-on-failure
```

| Suite | Binary | What it checks |
|---|---|---|
| Unit | `unit_tests` | Pure C++ logic: dtypes, tokenizer, stop strings, OpenAI JSON, SSE |
| Kernels | `kernel_tests` | Each CUDA kernel vs a naive CPU reference, across shapes |
| Golden | `golden_tests` | The full engine vs HF `transformers`, layer by layer |

Tests for unimplemented pieces are marked `GTEST_SKIP` and show as skipped, not failed.

### Tokenizer tests

`tests/data/tokenizer/<model>.json` holds ~350 strings (edge cases plus seeded random
unicode) with the ids HF `tokenizers` produces, and chat conversations rendered by
`apply_chat_template`. They need only the tokenizer files, not the weights:

```sh
tools/.venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct tokenizer.json tokenizer_config.json \
    generation_config.json config.json --local-dir models/qwen2.5-1.5b
tools/.venv/bin/hf download unsloth/Llama-3.2-1B-Instruct tokenizer.json tokenizer_config.json \
    generation_config.json config.json --local-dir models/llama3.2-1b
# regenerate a fixture (e.g. after changing the test strings):
tools/.venv/bin/python tools/dump_tokenizer_cases.py --model models/gpt2 --out tests/data/tokenizer/gpt2.json
```

### Golden tests

Golden tests catch the subtle bugs that make a model produce fluent nonsense: wrong RoPE
convention, wrong norm epsilon, Q/K/V fused in the wrong order, or a transposed weight.

```sh
tools/.venv/bin/python tools/dump_golden.py --model models/gpt2 --out golden/gpt2
./build/tests/golden_tests                    # uses golden/gpt2 by default

# any other model:
tools/.venv/bin/python tools/dump_golden.py --model models/qwen2.5-1.5b --out golden/qwen2.5-1.5b
OGRAPHY_GOLDEN_DIR=golden/qwen2.5-1.5b ./build/tests/golden_tests
```

The reference is HF in fp32 on the CPU. `LayerByLayer` compares the embeddings, the residual
stream after every layer, the final norm and the logits, and stops at the first stage whose
relative RMS error exceeds 1e-2 (fp16 lands around 1e-3). That stage points at the bug.
`GreedyMatchesHF` teacher-forces HF's greedy continuation through decode steps and requires
the same token at each step, except for near-ties. `Fp8KvCacheStaysClose` repeats that with
an FP8 KV cache, with looser checks (skipped for models with q/k/v biases).

For AWQ / GPTQ models, `dump_golden.py` dequantizes the checkpoint in plain torch and runs
that as the fp32 reference. `--check-against <fp16 model dir>` also checks the dequantized
model predicts like the original (Qwen2.5 1.5B: 99% next-token agreement for AWQ, 98% for
GPTQ); a misread packing format would give near zero:

```sh
tools/.venv/bin/python tools/dump_golden.py --model models/qwen2.5-1.5b-awq \
    --out golden/qwen2.5-1.5b-awq --check-against models/qwen2.5-1.5b
```

## Benchmarking

```sh
./build/apps/ography_bench --model models/qwen2.5-1.5b --prompt-len 512 --gen-len 256
./build/bench/bench_kernels
```

`ography_bench` feeds a random prompt (cold cache each iteration), generates greedily past
EOS, and reports the median time to first token, decode tokens/s and GPU memory in use.
`bench_kernels [filter]` times attention (prefill and decode) and the INT4 GEMMs on real
model shapes and reports TFLOPS or achieved bandwidth (peak ~256 GB/s).

Measured (`--gen-len 128`, cold cache):

| Model | TTFT, 512-token prompt | TTFT, 4096 | Decode (512 / 4096 context) | GPU memory |
|---|---|---|---|---|
| Llama 3.2 1B, fp16 | 33 ms | 303 ms | 95 / 90 tok/s | 2.7 GB |
| Qwen2.5 1.5B, AWQ | 61 ms | 471 ms | 162 / 155 tok/s | 1.4 GB |
| Qwen2.5 7B, AWQ | 303 ms | 2.3 s | 48 / 46 tok/s | 5.8 GB |

For deeper analysis:

```sh
nsys profile -o ography ./build/apps/ography_bench ...       # timeline: gaps, launch overhead
ncu --set full -k regex:rmsnorm ./build/bench/bench_kernels   # per-kernel roofline
```

## Memory planning on 8 GB

At startup the engine makes **one** allocation and splits it up:

```
[ weights | KV cache | activation scratch ]
```

KV cache size per token = `2 × layers × kv_heads × head_dim × bytes_per_element`.

Measured on the RTX 4070 Laptop (8 GB), default settings:

| Model | Weights on GPU | KV / token | Context that fits | Decode |
|---|---|---|---|---|
| Llama 3.2 1B Instruct, fp16 | 2.5 GB | 32 KB | 131K (model max) | 93 tok/s |
| Qwen2.5 1.5B Instruct, fp16 | 3.1 GB | 28 KB | 32K (model max) | 74 tok/s |
| Qwen2.5 1.5B Instruct, AWQ / GPTQ INT4 | 1.3 GB | 28 KB | 32K (model max) | 155 tok/s |
| Qwen2.5 7B Instruct, AWQ INT4 | 5.6 GB | 56 KB | ~22.8K | 48 tok/s |
| 8B (Llama 3.1), AWQ INT4 | ~5.7 GB | 128 KB (64 KB with `--fp8-kv`) | ~9K (~18K) | |
| 8B, fp16 | ~16 GB | | doesn't fit | |

At batch 1 decode reads every weight once per token, so INT4 is also ~2x faster than fp16.
`--fp8-kv` halves the KV cache, but costs accuracy on models with q/k/v biases (Qwen2): their
K vectors have outlier channels that FP8's ~6% precision can't hold, so the engine warns. It
is fine for Llama and GPT-2 (greedy output stays within near-ties of fp16).

## Roadmap

Progress is tracked here. The order matters: correctness before speed.

- [x] **1. Foundations:** `Tensor`, `DeviceArena`, safetensors loader, `ModelConfig`, GPT-2 weight mapping
  (Llama weight mapping comes with the Llama model)
- [x] **2. Kernels (GPT-2):** embedding + positions, LayerNorm, fused add+LayerNorm, bias+GELU,
  fused QKV split into the KV cache, naive causal attention (GQA-ready), cuBLAS GEMM, each with tests
- [x] **3. Forward pass (GPT-2):** `Gpt2Model::forward` with KV cache, prefill + decode, checked
  against a CPU fp32 reference
- [x] **3b. Llama:** RMSNorm, RoPE (default / linear / llama3 scaling, fused into the QKV split),
  SwiGLU, `map_llama_weights` (fused QKV and gate/up), `LlamaModel`. Llama 3.2 1B and Qwen2.5
  1.5B match HF layer by layer and greedy token for token
- [x] **4. Golden tests (GPT-2, Llama 3.2, Qwen2.5):** `dump_golden.py`, layer-by-layer agreement with HF, teacher-forced greedy
- [x] **5. Tokenizer:** byte-level BPE (GPT-2, Llama 3, Qwen2 pretokenizers), special tokens,
  ChatML / Llama 3 chat templates, incremental UTF-8 decoding; all matching HF exactly
  (NFC normalization and tool calls not yet)
- [x] **6. Generation:** GPU sampling (greedy, temperature, top-k/p, min-p, penalties), stop
  strings/tokens, `Engine::generate` with VRAM-sized KV cache and chunked prefill, `ography_cli`
- [x] **7. Server:** OpenAI chat / completions / models endpoints, SSE streaming, cancellation on
  disconnect, API key auth; checked with the official `openai` Python SDK
- [x] **8. Latency:** prefix reuse across turns (a follow-up question on a 5.6K-token document:
  TTFT 2.7 s -> 34 ms), CUDA-graph decode (captured once at load; ~1% at 1-2B where launches
  already overlap), fused add+RMSNorm
- [x] **9. Memory:** FP8 KV cache (`--fp8-kv`); INT4 AWQ and GPTQ checkpoints repacked at load
  into one layout, with a fused W4A16 GEMV for decode and L2-sized dequant tiles + cuBLAS for
  prefill. Qwen2.5 7B AWQ runs in 8 GB at 48 tok/s; the 1.5B AWQ/GPTQ models match their
  dequantized HF reference layer by layer
- [ ] **9b. Deferred:** FP8 weights (compressed-tensors), act-order GPTQ, calibrated FP8 KV scales
- [x] **10. Custom kernels:** tensor-core (WMMA) FlashAttention-2 prefill (15-28x the naive
  kernel, 20-25 TFLOPS), split-K decode attention with GQA grouping (5-6x, 150-200 GB/s),
  multi-token INT4 GEMV for prefills of <= 8 tokens. A 5.7K-token document on Qwen2.5 7B
  AWQ: prefill 8.5 s -> 3.3 s, decode at that length 23 -> 44 tok/s
- [ ] **10b. Next kernels:** a fused tensor-core W4A16 GEMM for 9-100-token INT4 prefills
  (now ~60 ms flat on 7B via dequant + cuBLAS); decode attention for large GQA groups is
  compute-bound (Qwen 7B: 150 vs 200 GB/s)
- [ ] **11. Stretch:** speculative decoding (draft model / n-gram), structured (JSON) output

## Conventions

- **Namespace** `ography`; kernels live in `ography::kernels`.
- **Includes** are rooted at `src/`: `#include "core/tensor.h"`.
- **One kernel per `.cu` file.** Every kernel gets a test in `tests/kernels/` and a benchmark in `bench/`.
- **Nothing allocates on the hot path.** All buffers come from the `DeviceArena` at startup.
  Decode-path code must be CUDA-graph capturable: no host syncs, no `cudaMalloc`, and
  per-step values read from device memory.
- **Tensors are views.** `Tensor` never owns memory. The arena or the mmapped file does.
- **Weight layout** follows HF, `[out_features, in_features]`, so `y = x @ Wᵀ`.
  Q/K/V and gate/up projections are fused at load time.
- **Unimplemented code** uses `OGRAPHY_TODO("what")`, which aborts with its location.
- **GPU errors are fatal:** wrap CUDA calls in `CUDA_CHECK`, cuBLAS calls in `CUBLAS_CHECK`,
  and follow launches with `CUDA_CHECK_LAUNCH()`.
- **Formatting:** `clang-format` with the repo's `.clang-format` (Google style, 100 columns).
