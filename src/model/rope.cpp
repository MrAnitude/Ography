#include "model/rope.h"

#include <cmath>
#include <stdexcept>

namespace ography {

std::vector<float> rope_inv_freq(const ModelConfig& c) {
  const int dim = c.head_dim;
  std::vector<float> inv(dim / 2);
  for (int i = 0; i < dim / 2; ++i) {
    // 1 / base^(2i / dim), in fp32 like torch.
    const float e = static_cast<float>(2 * i) / static_cast<float>(dim);
    inv[i] = 1.0f / std::pow(c.rope_theta, e);
  }

  const RopeScaling& rs = c.rope_scaling;
  if (rs.type.empty() || rs.type == "default") return inv;
  if (rs.type == "linear") {
    for (float& f : inv) f /= rs.factor;
    return inv;
  }
  if (rs.type == "llama3") {
    // transformers' _compute_llama3_parameters: long wavelengths are slowed by `factor`,
    // short ones kept, and the band in between interpolated.
    const float old_ctx = static_cast<float>(rs.original_max_position_embeddings);
    const float low_wavelen = old_ctx / rs.low_freq_factor;
    const float high_wavelen = old_ctx / rs.high_freq_factor;
    for (float& f : inv) {
      const float wavelen = 2.0f * static_cast<float>(M_PI) / f;
      if (wavelen > low_wavelen) {
        f = f / rs.factor;
      } else if (wavelen >= high_wavelen) {
        const float smooth =
            (old_ctx / wavelen - rs.low_freq_factor) / (rs.high_freq_factor - rs.low_freq_factor);
        f = (1.0f - smooth) * f / rs.factor + smooth * f;
      }
    }
    return inv;
  }
  throw std::runtime_error("unsupported rope_scaling type \"" + rs.type + "\"");
}

}  // namespace ography
