#include "engine/stop_checker.h"

#include "core/logging.h"

namespace ography {

StopChecker::StopChecker(const SamplingParams& params, std::vector<int32_t> eos_ids)
    : params_(params), eos_ids_(std::move(eos_ids)) {}

StopChecker::Result StopChecker::push(int32_t token_id, const std::string& text_piece,
                                      int tokens_generated) {
  OGRAPHY_TODO("StopChecker::push");
}

}  // namespace ography
