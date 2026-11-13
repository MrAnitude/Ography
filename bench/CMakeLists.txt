add_executable(bench_kernels bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE ography_kernels CUDA::cublas)
ography_set_flags(bench_kernels)
