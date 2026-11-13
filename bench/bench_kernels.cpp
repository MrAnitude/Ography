// Kernel microbenchmarks timed with cudaEvents (warmup, then median of N runs).
// Report achieved memory bandwidth (decode kernels are bandwidth-bound; compare against
// the 4070 Laptop's ~256 GB/s peak) and TFLOPS for prefill-shaped GEMMs.
//
// For deeper analysis: ncu --set full ./bench_kernels --kernel rmsnorm

#include "core/logging.h"
#include "kernels/kernels.h"

int main(int argc, char** argv) {
  using namespace ography;
  OGRAPHY_TODO("bench_kernels");
}
