#include "server/sse.h"

#include <nlohmann/json.hpp>

namespace ography::sse {

std::string event(const nlohmann::json& payload) { return "data: " + payload.dump() + "\n\n"; }

std::string done() { return "data: [DONE]\n\n"; }

}  // namespace ography::sse
