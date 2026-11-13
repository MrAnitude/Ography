#pragma once

#include <string>

#include <nlohmann/json_fwd.hpp>

namespace ography::sse {

// Server-sent events framing used by OpenAI streaming: "data: <json>\n\n", then "data: [DONE]\n\n".
std::string event(const nlohmann::json& payload);
std::string done();

}  // namespace ography::sse
