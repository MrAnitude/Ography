#pragma once

#include <string>
#include <vector>

namespace ography {

struct ChatMessage {
  std::string role;  // "system" | "user" | "assistant"
  std::string content;
};

// Renders a conversation into the prompt string the model was trained on.
// HF ships these as Jinja templates in tokenizer_config.json. Rather than interpret Jinja, the
// known formats are recognised from the template text and rendered by built-in code that
// reproduces them exactly (checked against transformers' apply_chat_template). A mini Jinja
// interpreter only if that stops being enough. Tool calls are not supported yet.
class ChatTemplate {
 public:
  // Reads "chat_template" (and "bos_token") from tokenizer_config.json. A missing file or
  // template gives the raw format (base models such as GPT-2). Throws for unknown templates.
  static ChatTemplate from_tokenizer_config(const std::string& tokenizer_config_path);
  static ChatTemplate from_template_text(const std::string& jinja, const std::string& bos_token);

  std::string render(const std::vector<ChatMessage>& messages,
                     bool add_generation_prompt = true) const;

  // True for models without a chat template: render() just concatenates message contents.
  bool is_raw() const { return format_ == Format::Raw; }
  const char* format_name() const;

  // Llama 3.2+ templates print today's date ("13 Nov 2026"); tests pin it.
  void set_date_string(std::string date) { date_string_ = std::move(date); }

 private:
  enum class Format {
    Raw,      // no template: contents concatenated
    ChatML,   // <|im_start|>role\n...<|im_end|>\n  (Qwen2/2.5); optional default system prompt
    Llama3,   // Llama 3.0: header + trimmed content + <|eot_id|>, BOS before the first message
    Llama31,  // Llama 3.1+: always a system block with "Cutting Knowledge Date" and a date
  };
  Format format_ = Format::Raw;
  std::string bos_token_;
  std::string default_system_;  // ChatML: used when the conversation has no system message
  std::string date_string_;     // Llama31
};

}  // namespace ography
