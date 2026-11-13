#include <gtest/gtest.h>

#include "kernels/kernels.h"

// TODO: random fp16 input -> kernels::rmsnorm on GPU vs fp32 CPU reference, max abs err < 1e-2.
//       Same for add_rmsnorm (check residual is updated too). Sweep hidden sizes 2048..8192.
TEST(Kernels, RmsNorm) { GTEST_SKIP() << "kernel not implemented yet"; }
