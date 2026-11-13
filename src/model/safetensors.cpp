#include "model/safetensors.h"

#include "core/logging.h"

namespace ography {

SafetensorsFile::SafetensorsFile(const std::string& path) {
  OGRAPHY_TODO("SafetensorsFile: mmap + parse JSON header");
}

SafetensorsFile::~SafetensorsFile() {
  // TODO: munmap(mapping_, mapping_size_) once the constructor maps the file.
}

bool SafetensorsFile::contains(const std::string& name) const { return tensors_.count(name) > 0; }

Tensor SafetensorsFile::get(const std::string& name) const {
  OGRAPHY_TODO("SafetensorsFile::get");
}

std::vector<std::string> SafetensorsFile::names() const {
  OGRAPHY_TODO("SafetensorsFile::names");
}

SafetensorsModel::SafetensorsModel(const std::string& model_dir) {
  OGRAPHY_TODO("SafetensorsModel: open single file or shards from index.json");
}

bool SafetensorsModel::contains(const std::string& name) const {
  return name_to_file_.count(name) > 0;
}

Tensor SafetensorsModel::get(const std::string& name) const {
  OGRAPHY_TODO("SafetensorsModel::get");
}

}  // namespace ography
