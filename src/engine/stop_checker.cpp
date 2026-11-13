#include "engine/stop_checker.h"

#include <algorithm>

namespace ography {

StopChecker::StopChecker(const SamplingParams& params, std::vector<int32_t> eos_ids)
    : params_(params), stop_ids_(std::move(eos_ids)) {
  stop_ids_.insert(stop_ids_.end(), params.stop_token_ids.begin(), params.stop_token_ids.end());
  // An empty stop string would match immediately; ignore it as OpenAI does.
  auto& stop = params_.stop;
  stop.erase(std::remove(stop.begin(), stop.end(), std::string()), stop.end());
}

bool StopChecker::is_stop_token(int32_t token_id) const {
  return std::find(stop_ids_.begin(), stop_ids_.end(), token_id) != stop_ids_.end();
}

StopChecker::Result StopChecker::push(int32_t token_id, const std::string& text_piece,
                                      int tokens_generated) {
  Result r;
  held_back_ += text_piece;

  // A full stop string: emit what precedes the earliest match and finish.
  size_t first = std::string::npos;
  for (const auto& s : params_.stop) first = std::min(first, held_back_.find(s));
  if (first != std::string::npos) {
    r.emit = held_back_.substr(0, first);
    held_back_.clear();
    r.finish = FinishReason::Stop;
    return r;
  }

  if (is_stop_token(token_id) || tokens_generated >= params_.max_tokens) {
    r.emit = std::move(held_back_);
    held_back_.clear();
    r.finish = is_stop_token(token_id) ? FinishReason::Stop : FinishReason::Length;
    return r;
  }

  // Hold back the longest suffix that is a proper prefix of some stop string.
  size_t keep = 0;
  for (const auto& s : params_.stop) {
    for (size_t len = std::min(s.size() - 1, held_back_.size()); len > keep; --len) {
      if (held_back_.compare(held_back_.size() - len, len, s, 0, len) == 0) {
        keep = len;
        break;
      }
    }
  }
  r.emit = held_back_.substr(0, held_back_.size() - keep);
  held_back_.erase(0, held_back_.size() - keep);
  return r;
}

}  // namespace ography
