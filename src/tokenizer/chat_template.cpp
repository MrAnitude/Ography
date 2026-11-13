#include "tokenizer/chat_template.h"

#include <nlohmann/json.hpp>

#include "core/logging.h"

namespace ography {

ChatTemplate ChatTemplate::from_tokenizer_config(const std::string& tokenizer_config_path) {
  OGRAPHY_TODO("ChatTemplate::from_tokenizer_config");
}

std::string ChatTemplate::render(const std::vector<ChatMessage>& messages,
                                 bool add_generation_prompt) const {
  OGRAPHY_TODO("ChatTemplate::render");
}

}  // namespace ography
