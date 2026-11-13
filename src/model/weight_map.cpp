#include "model/weight_map.h"

#include "core/logging.h"

namespace ography {

ModelWeights map_weights(const SafetensorsModel& checkpoint, const ModelConfig& config,
                         DeviceArena& arena) {
  OGRAPHY_TODO("map_weights: name table, fuse qkv / gate_up, upload to GPU");
}

}  // namespace ography
