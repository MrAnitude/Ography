#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "core/cuda_check.h"
#include "core/memory.h"

using namespace ography;

TEST(DeviceArena, AllocatesAlignedAndWritable) {
  DeviceArena arena(1 << 20);
  Tensor a = arena.allocate(DType::F16, {3});   // 6 bytes
  Tensor b = arena.allocate(DType::F32, {100});
  EXPECT_EQ(a.device(), Device::CUDA);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(a.data()) % 256, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(b.data()) % 256, 0u);
  EXPECT_EQ(static_cast<char*>(b.data()) - static_cast<char*>(a.data()), 256);
  EXPECT_EQ(arena.used(), 256u + 400u);

  std::vector<float> host(100), back(100);
  for (int i = 0; i < 100; ++i) host[i] = static_cast<float>(i);
  CUDA_CHECK(cudaMemcpy(b.data(), host.data(), 400, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(back.data(), b.data(), 400, cudaMemcpyDeviceToHost));
  EXPECT_EQ(host, back);
}

TEST(DeviceArena, MarkAndRewindReuseMemory) {
  DeviceArena arena(1 << 20);
  arena.allocate(DType::F32, {10});
  const size_t m = arena.mark();
  Tensor scratch1 = arena.allocate(DType::F32, {1000});
  arena.rewind(m);
  Tensor scratch2 = arena.allocate(DType::F32, {1000});
  EXPECT_EQ(scratch1.data(), scratch2.data());
}

TEST(DeviceArenaDeathTest, AbortsWhenExhausted) {
  EXPECT_DEATH(
      {
        DeviceArena arena(1024);
        arena.allocate(DType::F32, {1024});
      },
      "DeviceArena exhausted");
}

TEST(DeviceArena, QueryFreeMemory) {
  EXPECT_GT(query_free_device_memory(), 0u);
}
