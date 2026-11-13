#include "tokenizer/chat_template.h"

#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "tokenizer/unicode.h"

namespace ography {

namespace {

bool contains(const std::string& s, const char* what) { return s.find(what) != std::string::npos; }

// Jinja's |trim, i.e. Python str.strip(): Unicode White_Space plus U+001C..U+001F.
std::string trim(const std::string& s) {
  auto is_space = [](char32_t c) { return unicode::is_whitespace(c) || (c >= 0x1C && c <= 0x1F); };
  size_t begin = 0, end = 0;  // [begin, end) of the non-space span
  bool seen = false;
  for (size_t i = 0; i < s.size();) {
    char32_t cp;
    const int len = unicode::decode_utf8(s, i, cp);
    if (!is_space(cp)) {
      if (!seen) begin = i;
      seen = true;
      end = i + len;
    }
    i += len;
  }
  return seen ? s.substr(begin, end - begin) : std::string();
}

// The ChatML default system prompt, if the template has one: the string literal
// '<|im_start|>system\n<default><|im_end|>\n' (Qwen2.5: "You are Qwen, created by ...").
std::string chatml_default_system(const std::string& jinja) {
  const std::string open = "'<|im_start|>system\\n";
  const std::string close = "<|im_end|>\\n'";
  for (size_t pos = jinja.find(open); pos != std::string::npos; pos = jinja.find(open, pos + 1)) {
    const size_t start = pos + open.size();
    const size_t quote = jinja.find('\'', start);
    if (quote == std::string::npos) break;
    // A literal that runs to <|im_end|>\n without concatenation is the default prompt.
    if (quote + 1 >= close.size() && jinja.compare(quote + 1 - close.size(), close.size(), close) == 0) {
      std::string text = jinja.substr(start, quote + 1 - close.size() - start);
      if (text.find('\\') != std::string::npos) {
        throw std::runtime_error("chat template: escapes in the default system prompt");
      }
      return text;
    }
  }
  return "";
}

std::string today() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%d %b %Y", &tm);
  return buf;
}

void check_role(const ChatMessage& m) {
  if (m.role != "system" && m.role != "user" && m.role != "assistant") {
    throw std::invalid_argument("chat template: unsupported role \"" + m.role + "\"");
  }
}

}  // namespace

ChatTemplate ChatTemplate::from_tokenizer_config(const std::string& path) {
  if (!std::filesystem::exists(path)) return ChatTemplate();
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  const auto config = nlohmann::json::parse(f);
  if (!config.contains("chat_template") || !config["chat_template"].is_string()) {
    return ChatTemplate();  // absent, or a list of named templates (not supported yet)
  }
  std::string bos;
  if (config.contains("bos_token")) {
    const auto& b = config["bos_token"];
    if (b.is_string()) bos = b.get<std::string>();
    if (b.is_object()) bos = b.value("content", "");
  }
  return from_template_text(config["chat_template"].get<std::string>(), bos);
}

ChatTemplate ChatTemplate::from_template_text(const std::string& jinja, const std::string& bos) {
  ChatTemplate t;
  t.bos_token_ = bos;
  if (contains(jinja, "<|im_start|>")) {
    t.format_ = Format::ChatML;
    t.default_system_ = chatml_default_system(jinja);
  } else if (contains(jinja, "<|start_header_id|>")) {
    if (contains(jinja, "Cutting Knowledge Date")) {
      t.format_ = Format::Llama31;
      // 3.2+ print today's date via strftime_now; 3.1 hard-codes this one.
      t.date_string_ = contains(jinja, "strftime_now") ? today() : "26 Jul 2024";
    } else {
      t.format_ = Format::Llama3;
    }
  } else {
    throw std::runtime_error("chat template: unrecognised format (supported: ChatML, Llama 3)");
  }
  return t;
}

const char* ChatTemplate::format_name() const {
  switch (format_) {
    case Format::Raw: return "raw";
    case Format::ChatML: return "chatml";
    case Format::Llama3: return "llama3";
    case Format::Llama31: return "llama3.1";
  }
  return "?";
}

std::string ChatTemplate::render(const std::vector<ChatMessage>& messages,
                                 bool add_generation_prompt) const {
  for (const auto& m : messages) check_role(m);
  std::string out;

  switch (format_) {
    case Format::Raw:
      for (const auto& m : messages) out += m.content;
      return out;

    case Format::ChatML: {
      if (messages.empty()) throw std::invalid_argument("chat template: no messages");
      size_t first = 0;
      if (messages[0].role == "system") {
        out += "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n";
        first = 1;
      } else if (!default_system_.empty()) {
        out += "<|im_start|>system\n" + default_system_ + "<|im_end|>\n";
      }
      for (size_t i = first; i < messages.size(); ++i) {
        out += "<|im_start|>" + messages[i].role + "\n" + messages[i].content + "<|im_end|>\n";
      }
      if (add_generation_prompt) out += "<|im_start|>assistant\n";
      return out;
    }

    case Format::Llama3:
      out += bos_token_;
      for (const auto& m : messages) {
        out += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + trim(m.content) +
               "<|eot_id|>";
      }
      if (add_generation_prompt) out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
      return out;

    case Format::Llama31: {
      size_t first = 0;
      std::string system;
      if (!messages.empty() && messages[0].role == "system") {
        system = trim(messages[0].content);
        first = 1;
      }
      out += bos_token_;
      out += "<|start_header_id|>system<|end_header_id|>\n\n";
      out += "Cutting Knowledge Date: December 2023\n";
      out += "Today Date: " + date_string_ + "\n\n";
      out += system + "<|eot_id|>";
      for (size_t i = first; i < messages.size(); ++i) {
        out += "<|start_header_id|>" + messages[i].role + "<|end_header_id|>\n\n" +
               trim(messages[i].content) + "<|eot_id|>";
      }
      if (add_generation_prompt) out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
      return out;
    }
  }
  return out;
}

}  // namespace ography
