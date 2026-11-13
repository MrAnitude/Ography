#include "tokenizer/bpe.h"

#include <nlohmann/json.hpp>

#include "core/logging.h"

namespace ography {

BpeModel::BpeModel(const nlohmann::json& tokenizer_json) {
  OGRAPHY_TODO("BpeModel: load vocab, merges, added_tokens");
}

void BpeModel::encode_ordinary(std::string_view text, std::vector<int32_t>& out) const {
  OGRAPHY_TODO("BpeModel::encode_ordinary");
}

const std::string& BpeModel::token_bytes(int32_t id) const { return id_to_bytes_.at(id); }

}  // namespace ography
