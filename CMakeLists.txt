cmake_minimum_required(VERSION 3.24)

# Target the RTX 4070 (Ada Lovelace, sm_89) unless overridden on the command line.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(ography VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(OGRAPHY_BUILD_TESTS "Build unit, kernel and golden tests" ON)
option(OGRAPHY_BUILD_BENCH "Build kernel microbenchmarks" ON)

find_package(CUDAToolkit REQUIRED)

include(cmake/CompilerFlags.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)
add_subdirectory(apps)

if(OGRAPHY_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(OGRAPHY_BUILD_BENCH)
  add_subdirectory(bench)
endif()
