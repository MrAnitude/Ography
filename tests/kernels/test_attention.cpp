#include <gtest/gtest.h>

#include "kernels/kernels.h"

// TODO: naive CPU softmax(QK^T * scale + causal mask) V reference.
//       Prefill: various T, start_pos > 0 (prefix reuse), GQA ratios 1/4/8.
//       Decode: seq_len 1, 17, 4096 (exercises split-K reduction).
TEST(Kernels, AttentionPrefill) { GTEST_SKIP() << "kernel not implemented yet"; }
TEST(Kernels, AttentionDecode) { GTEST_SKIP() << "kernel not implemented yet"; }
