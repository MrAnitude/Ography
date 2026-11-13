#include "model/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "core/logging.h"

namespace ography {

namespace fs = std::filesystem;

SafetensorsFile::SafetensorsFile(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));

  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  mapping_size_ = static_cast<size_t>(st.st_size);
  if (mapping_size_ < 8) {
    ::close(fd);
    throw std::runtime_error(path + ": too small to be a safetensors file");
  }
  // Read-only, private mapping: tensors returned by get() must never be written through.
  mapping_ = ::mmap(nullptr, mapping_size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (mapping_ == MAP_FAILED) {
    mapping_ = nullptr;
    throw std::runtime_error("cannot mmap " + path + ": " + std::strerror(errno));
  }

  // The destructor doesn't run if the constructor throws, so unmap here on any parse error.
  try {
    const auto* bytes = static_cast<const uint8_t*>(mapping_);
    uint64_t header_size = 0;
    for (int i = 7; i >= 0; --i) header_size = (header_size << 8) | bytes[i];  // little-endian
    if (header_size > mapping_size_ - 8) {
      throw std::runtime_error(path + ": header size " + std::to_string(header_size) +
                               " exceeds file size");
    }

    nlohmann::json header;
    try {
      header = nlohmann::json::parse(bytes + 8, bytes + 8 + header_size);
    } catch (const nlohmann::json::exception& e) {
      throw std::runtime_error(path + ": bad JSON header: " + e.what());
    }

    const uint8_t* data_start = bytes + 8 + header_size;
    const size_t data_size = mapping_size_ - 8 - header_size;

    for (const auto& [name, info] : header.items()) {
      if (name == "__metadata__") continue;

      const std::string dtype_str = info.at("dtype").get<std::string>();
      DType dtype;
      try {
        dtype = dtype_from_safetensors(dtype_str);
      } catch (const std::invalid_argument&) {
        // Rare buffers (F64, BOOL, ...) shouldn't stop the rest of the checkpoint loading.
        log(LogLevel::Warn, path + ": skipping " + name + " with unsupported dtype " + dtype_str);
        continue;
      }

      auto shape = info.at("shape").get<std::vector<int64_t>>();
      const auto offsets = info.at("data_offsets").get<std::vector<uint64_t>>();
      if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > data_size) {
        throw std::runtime_error(path + ": bad data_offsets for " + name);
      }

      Tensor t(const_cast<uint8_t*>(data_start + offsets[0]), dtype, std::move(shape),
               Device::CPU);
      if (t.nbytes() != offsets[1] - offsets[0]) {
        throw std::runtime_error(path + ": " + name + " has " +
                                 std::to_string(offsets[1] - offsets[0]) +
                                 " bytes but its shape needs " + std::to_string(t.nbytes()));
      }
      tensors_.emplace(name, std::move(t));
    }
  } catch (...) {
    ::munmap(mapping_, mapping_size_);
    mapping_ = nullptr;
    throw;
  }
}

SafetensorsFile::~SafetensorsFile() {
  if (mapping_ != nullptr) ::munmap(mapping_, mapping_size_);
}

bool SafetensorsFile::contains(const std::string& name) const { return tensors_.count(name) > 0; }

Tensor SafetensorsFile::get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::out_of_range("tensor not found: " + name);
  return it->second;
}

std::vector<std::string> SafetensorsFile::names() const {
  std::vector<std::string> out;
  out.reserve(tensors_.size());
  for (const auto& [name, _] : tensors_) out.push_back(name);
  return out;
}

SafetensorsModel::SafetensorsModel(const std::string& model_dir) {
  const fs::path dir(model_dir);
  std::vector<fs::path> paths;

  const fs::path index = dir / "model.safetensors.index.json";
  if (fs::exists(index)) {
    std::ifstream in(index);
    const auto j = nlohmann::json::parse(in);
    std::vector<std::string> shard_names;
    for (const auto& [_, file] : j.at("weight_map").items()) {
      const auto f = file.get<std::string>();
      if (std::find(shard_names.begin(), shard_names.end(), f) == shard_names.end()) {
        shard_names.push_back(f);
      }
    }
    std::sort(shard_names.begin(), shard_names.end());
    for (const auto& f : shard_names) paths.push_back(dir / f);
  } else if (fs::exists(dir / "model.safetensors")) {
    paths.push_back(dir / "model.safetensors");
  } else {
    throw std::runtime_error(model_dir +
                             ": no model.safetensors or model.safetensors.index.json found");
  }

  for (const auto& p : paths) {
    files_.push_back(std::make_unique<SafetensorsFile>(p.string()));
    const size_t idx = files_.size() - 1;
    for (const auto& name : files_.back()->names()) name_to_file_.emplace(name, idx);
  }
}

bool SafetensorsModel::contains(const std::string& name) const {
  return name_to_file_.count(name) > 0;
}

Tensor SafetensorsModel::get(const std::string& name) const {
  auto it = name_to_file_.find(name);
  if (it == name_to_file_.end()) throw std::out_of_range("tensor not found in checkpoint: " + name);
  return files_[it->second]->get(name);
}

}  // namespace ography
