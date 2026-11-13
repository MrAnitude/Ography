// End-to-end latency benchmark.
//
//   ography_bench --model /path/to/hf-model --prompt-len 512 --gen-len 256 --iters 5
//
// Reports time-to-first-token (prefill), time per output token (decode), tokens/sec,
// and peak VRAM, so every optimisation can be measured against a baseline.

#include "core/logging.h"
#include "engine/engine.h"

int main(int argc, char** argv) {
  using namespace ography;
  OGRAPHY_TODO("ography_bench");
}
