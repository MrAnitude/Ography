#pragma once

#include <vector>

#include "model/config.h"

namespace ography {

// RoPE inverse frequencies [head_dim / 2] for a config, with rope_scaling applied, computed
// the way HF transformers does (fp32). Supports default, "linear" and "llama3" scaling;
// throws for others (yarn, dynamic, longrope).
std::vector<float> rope_inv_freq(const ModelConfig& config);

}  // namespace ography
