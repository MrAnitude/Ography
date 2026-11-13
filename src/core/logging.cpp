#include "core/logging.h"

#include <cstdio>
#include <cstdlib>

namespace ography {

namespace {
LogLevel g_level = LogLevel::Info;

const char* level_tag(LogLevel level) {
  switch (level) {
    case LogLevel::Debug: return "DEBUG";
    case LogLevel::Info: return "INFO";
    case LogLevel::Warn: return "WARN";
    case LogLevel::Error: return "ERROR";
  }
  return "?";
}
}  // namespace

void set_log_level(LogLevel level) { g_level = level; }

void log(LogLevel level, std::string_view msg) {
  if (level < g_level) return;
  std::fprintf(stderr, "[%s] %.*s\n", level_tag(level), static_cast<int>(msg.size()), msg.data());
}

void todo_abort(std::string_view what, const char* file, int line) {
  std::fprintf(stderr, "TODO not implemented: %.*s (%s:%d)\n", static_cast<int>(what.size()),
               what.data(), file, line);
  std::abort();
}

}  // namespace ography
