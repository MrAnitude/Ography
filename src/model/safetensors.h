#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/tensor.h"

namespace ography {

// One mmapped .safetensors file. Tensors returned are CPU views into the mapping.
// Format: 8-byte little-endian header size, JSON header, then raw tensor bytes.
class SafetensorsFile {
 public:
  explicit SafetensorsFile(const std::string& path);
  ~SafetensorsFile();
  SafetensorsFile(const SafetensorsFile&) = delete;
  SafetensorsFile& operator=(const SafetensorsFile&) = delete;

  bool contains(const std::string& name) const;
  Tensor get(const std::string& name) const;
  std::vector<std::string> names() const;

 private:
  void* mapping_ = nullptr;
  size_t mapping_size_ = 0;
  std::unordered_map<std::string, Tensor> tensors_;
};

// A whole HF model directory: a single model.safetensors, or shards listed in
// model.safetensors.index.json.
class SafetensorsModel {
 public:
  explicit SafetensorsModel(const std::string& model_dir);

  bool contains(const std::string& name) const;
  Tensor get(const std::string& name) const;  // throws if missing
  std::vector<std::string> names() const;

 private:
  std::vector<std::unique_ptr<SafetensorsFile>> files_;
  std::unordered_map<std::string, size_t> name_to_file_;
};

}  // namespace ography
