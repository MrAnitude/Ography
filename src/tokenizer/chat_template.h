#pragma once

#include <string>
#include <vector>

namespace ography {

struct ChatMessage {
  std::string role;  // "system" | "user" | "assistant"
  std::string content;
};

// Renders a conversation into the prompt string the model was trained on.
// HF ships these as Jinja templates in tokenizer_config.json. Plan: built-in renderers
// for known formats first (llama3, chatml, mistral), detected from the template text;
// a mini Jinja interpreter only if that stops being enough.
class ChatTemplate {
 public:
  static ChatTemplate from_tokenizer_config(const std::string& tokenizer_config_path);

  std::string render(const std::vector<ChatMessage>& messages,
                     bool add_generation_prompt = true) const;

 private:
  enum class Format { Llama3, ChatML, Mistral };
  Format format_ = Format::ChatML;
  std::string bos_token_;
};

}  // namespace ography
