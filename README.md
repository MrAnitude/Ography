# Ography

A from-scratch LLM inference engine written in C++ and CUDA, built for **low-latency,
single-user inference** on a consumer laptop GPU. It loads Hugging Face checkpoints
(`config.json` + `*.safetensors` + `tokenizer.json`), runs the forward pass on custom CUDA
kernels, and serves the model through an **OpenAI-compatible HTTP API** so existing
OpenAI clients and tools work against it unchanged.

> **Status: scaffold.** The module layout, build system, interfaces and test harness are in
> place. Most function bodies are `OGRAPHY_TODO(...)` stubs that abort with their file and
> line if called. See the [roadmap](#roadmap) for build order and progress.

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
- Run off-the-shelf Hugging Face decoder models (Llama-family first: Llama, Mistral, Qwen2).
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
| Toolchain | CUDA 13.2, GCC 13, CMake ≥ 3.24 |

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
   (reuse KV from the          (LlamaModel: embedding →        (penalties, temperature,
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
│   │   └── arch/               # model architectures (llama.{h,cpp})
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
│   └── golden/                 # engine vs HF transformers, layer by layer
├── bench/                      # kernel microbenchmarks (cudaEvent timing)
├── tools/                      # Python, offline only
│   ├── dump_golden.py          # dump HF reference tensors for tests/golden
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
| `kernels` | `kernels::rmsnorm`, `attention_decode`, `gemm`, … | All GPU compute. Async, stream-ordered, CUDA-graph safe |
| `kv_cache` | `KVCache`, `PrefixTracker` | Per-layer K/V storage; reusing cached prompt prefixes |
| `quant` | `QuantConfig`, `QuantizedWeight` | Describes quantized checkpoint formats |
| `model` | `ModelConfig`, `SafetensorsModel`, `ModelWeights`, `Model` | Loading, mapping and running a transformer |
| `tokenizer` | `Tokenizer`, `IncrementalDecoder`, `ChatTemplate` | Text ↔ token ids, prompt formatting |
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
export PATH=/usr/local/cuda-13.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:$LD_LIBRARY_PATH
```

Verify with `nvcc --version`.

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
tools/.venv/bin/huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/qwen2.5-1.5b
tools/.venv/bin/python tools/inspect_safetensors.py models/qwen2.5-1.5b
```

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

If CMake can't find `nvcc`, pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc`.

`compile_commands.json` is generated in `build/` for clangd and other IDE tooling.

## Running

> The commands below describe the intended interface. They abort with a `TODO` message
> until the engine is implemented.

### Server

```sh
./build/apps/ography_server --model models/qwen2.5-1.5b --port 8000
```

| Flag | Meaning |
|---|---|
| `--model DIR` | Hugging Face model directory |
| `--host`, `--port` | Bind address (defaults to `127.0.0.1:8000`, so it's local only) |
| `--max-seq-len N` | Context length; default is as large as free VRAM allows |
| `--fp8-kv` | Store the KV cache in FP8 (half the memory, so twice the context) |
| `--no-cuda-graphs` | Disable decode graph capture (for debugging) |
| `--api-key KEY` | Require `Authorization: Bearer KEY` |

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health`.

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
./build/apps/ography_cli --model models/qwen2.5-1.5b --temperature 0.7
```

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

### Golden tests

Golden tests catch the subtle bugs that make a model produce fluent nonsense: wrong RoPE
convention, wrong norm epsilon, Q/K/V fused in the wrong order, or a transposed weight.

```sh
tools/.venv/bin/python tools/dump_golden.py --model models/qwen2.5-1.5b --out golden/qwen2.5-1.5b
OGRAPHY_GOLDEN_DIR=golden/qwen2.5-1.5b ./build/tests/golden_tests
```

The first layer whose hidden state diverges from the reference points at the bug.

## Benchmarking

```sh
./build/apps/ography_bench --model models/qwen2.5-1.5b --prompt-len 512 --gen-len 256
./build/bench/bench_kernels
```

`ography_bench` reports time-to-first-token, time per output token, tokens/s and peak VRAM.
`bench_kernels` reports achieved bandwidth per kernel. Compare it against the ~256 GB/s
peak to see how close each decode kernel is to its limit.

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

| Model (example) | Weights | KV / token (fp16) | Fits in 8 GB? |
|---|---|---|---|
| 1.5B, fp16 | ~3.1 GB | ~28 KB | Yes, with long context |
| 3B, fp16 | ~6.4 GB | ~36 KB | Tight: short context |
| 8B, INT4 (AWQ/GPTQ) | ~4.5–5 GB | 128 KB (64 KB with FP8 KV) | Yes, ~16–24K context |
| 8B, fp16 | ~16 GB | n/a | No: needs quantization |

## Roadmap

Progress is tracked here. The order matters: correctness before speed.

- [ ] **1. Foundations:** `Tensor`, `DeviceArena`, safetensors loader, `ModelConfig`, weight mapping
- [ ] **2. Kernels:** embedding, RMSNorm, RoPE, SwiGLU, attention (naive first), cuBLAS GEMM, each with tests
- [ ] **3. Forward pass:** `LlamaModel::forward` with KV cache, prefill + decode
- [ ] **4. Golden tests:** `dump_golden.py`, then layer-by-layer agreement with HF
- [ ] **5. Tokenizer:** byte-level BPE, special tokens, chat templates, incremental decoding
- [ ] **6. Generation:** sampling kernels, stop handling, `Engine::generate`, `ography_cli`
- [ ] **7. Server:** OpenAI endpoints, SSE streaming, cancellation on disconnect
- [ ] **8. Latency:** prefix reuse across turns, CUDA-graph decode, fused add+RMSNorm
- [ ] **9. Memory:** FP8 KV cache, INT4 weight loading (AWQ/GPTQ), FP8 weights
- [ ] **10. Custom kernels:** FlashAttention-style prefill, split-K decode attention, W4A16 GEMV replacing cuBLAS for decode
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
