#pragma once

#include <string_view>

namespace ography {

enum class LogLevel { Debug, Info, Warn, Error };

void set_log_level(LogLevel level);
void log(LogLevel level, std::string_view msg);

// Marks unimplemented code. Prints where and aborts, so a stub can never fail silently.
[[noreturn]] void todo_abort(std::string_view what, const char* file, int line);

}  // namespace ography

#define OGRAPHY_TODO(what) ::ography::todo_abort((what), __FILE__, __LINE__)
