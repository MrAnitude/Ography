#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "model/safetensors.h"

using namespace ography;
namespace fs = std::filesystem;

namespace {

struct Entry {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

void write_safetensors(const fs::path& path, const std::vector<Entry>& entries) {
  nlohmann::json header;
  header["__metadata__"] = {{"format", "pt"}};
  uint64_t offset = 0;
  for (const auto& e : entries) {
    header[e.name] = {{"dtype", e.dtype},
                      {"shape", e.shape},
                      {"data_offsets", {offset, offset + e.bytes.size()}}};
    offset += e.bytes.size();
  }
  const std::string h = header.dump();
  const uint64_t n = h.size();
  std::ofstream out(path, std::ios::binary);
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xff));
  out << h;
  for (const auto& e : entries) out.write(reinterpret_cast<const char*>(e.bytes.data()), e.bytes.size());
}

template <typename T>
std::vector<uint8_t> as_bytes(std::vector<T> v) {
  std::vector<uint8_t> b(v.size() * sizeof(T));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

class SafetensorsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::path(::testing::TempDir()) /
           ("ography_st_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }
  fs::path dir_;
};

}  // namespace

TEST_F(SafetensorsTest, ReadsTensors) {
  write_safetensors(dir_ / "model.safetensors",
                    {{"a", "F32", {2, 3}, as_bytes<float>({0, 1, 2, 3, 4, 5})},
                     {"b", "I32", {2}, as_bytes<int32_t>({7, -7})}});
  SafetensorsFile f((dir_ / "model.safetensors").string());

  EXPECT_TRUE(f.contains("a"));
  EXPECT_FALSE(f.contains("__metadata__"));
  EXPECT_EQ(f.names().size(), 2u);

  Tensor a = f.get("a");
  EXPECT_EQ(a.dtype(), DType::F32);
  EXPECT_EQ(a.device(), Device::CPU);
  EXPECT_EQ(a.shape(), (std::vector<int64_t>{2, 3}));
  EXPECT_FLOAT_EQ(a.data_as<float>()[5], 5.0f);

  Tensor b = f.get("b");
  EXPECT_EQ(b.data_as<int32_t>()[1], -7);
  EXPECT_THROW(f.get("missing"), std::out_of_range);
}

TEST_F(SafetensorsTest, SkipsUnsupportedDtypes) {
  write_safetensors(dir_ / "model.safetensors",
                    {{"keep", "F16", {1}, {0, 0x3c}}, {"skip", "F64", {1}, std::vector<uint8_t>(8)}});
  SafetensorsFile f((dir_ / "model.safetensors").string());
  EXPECT_TRUE(f.contains("keep"));
  EXPECT_FALSE(f.contains("skip"));
}

TEST_F(SafetensorsTest, RejectsSizeMismatch) {
  write_safetensors(dir_ / "bad.safetensors", {{"a", "F32", {4}, as_bytes<float>({1, 2})}});
  EXPECT_THROW(SafetensorsFile((dir_ / "bad.safetensors").string()), std::runtime_error);
}

TEST_F(SafetensorsTest, RejectsTruncatedHeader) {
  std::ofstream(dir_ / "trunc.safetensors", std::ios::binary) << std::string("\xff\x00\x00\x00\x00\x00\x00\x00{", 9);
  EXPECT_THROW(SafetensorsFile((dir_ / "trunc.safetensors").string()), std::runtime_error);
}

TEST_F(SafetensorsTest, ModelFromShards) {
  write_safetensors(dir_ / "model-00001-of-00002.safetensors", {{"x", "F32", {1}, as_bytes<float>({1})}});
  write_safetensors(dir_ / "model-00002-of-00002.safetensors", {{"y", "F32", {1}, as_bytes<float>({2})}});
  nlohmann::json index = {{"weight_map",
                           {{"x", "model-00001-of-00002.safetensors"},
                            {"y", "model-00002-of-00002.safetensors"}}}};
  std::ofstream(dir_ / "model.safetensors.index.json") << index.dump();

  SafetensorsModel m(dir_.string());
  EXPECT_FLOAT_EQ(m.get("x").data_as<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(m.get("y").data_as<float>()[0], 2.0f);
  EXPECT_THROW(m.get("z"), std::out_of_range);
}

TEST_F(SafetensorsTest, ModelMissingFiles) {
  EXPECT_THROW(SafetensorsModel(dir_.string()), std::runtime_error);
}
