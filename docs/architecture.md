# Architecture

## Constraints that shape the design

- **One client, one request at a time.** No continuous batching, no paged KV cache.
  Everything is optimised for batch size 1 latency.
- **8 GB VRAM.** Memory is planned once at startup: weights, then the KV cache sized from
  what's left, then fixed activation scratch. Nothing is allocated on the hot path.
- **Decode is memory-bandwidth bound.** At batch 1 each token reads every weight once, so
  tokens/sec ≈ bandwidth / model bytes. Quantized weights (INT4/FP8) are the main lever;
  kernel launch overhead (fixed with CUDA graphs) is the second.

## Module dependencies

```
core <- kernels <- kv_cache <- model <- engine <- server
core <- quant   <- model
tokenizer (CPU only) <- engine
```

## Request flow

1. `server/` receives `POST /v1/chat/completions`, parses it into messages + `SamplingParams`.
2. `ChatTemplate::render` produces the prompt; `Tokenizer::encode` turns it into ids.
3. The job goes to `RequestQueue`, whose single worker thread owns the `Engine`.
4. `Engine::generate`:
   - `PrefixTracker` finds how much of the prompt is already in the KV cache
     (the whole previous conversation, in multi-turn chat) and truncates the cache to it.
   - **Prefill** the remaining prompt tokens in chunks of `prefill_chunk`.
   - **Decode** loop: replay the captured CUDA graph (forward + sample) for one token, copy
     the single sampled id to the host, `IncrementalDecoder` → `StopChecker` → callback.
5. The callback streams each piece as an SSE chunk; a disconnect returns `false` and
   cancels generation.

## Memory plan (startup)

```
[ weights | KV cache (max_seq_len from free VRAM) | activation scratch ]
           one DeviceArena, one cudaMalloc
```

KV bytes per token = 2 × layers × kv_heads × head_dim × dtype size.
For example, an 8B Llama 3 model (32 layers, 8 KV heads, head_dim 128) needs 128 KiB/token
in fp16, or 64 KiB/token in FP8.
