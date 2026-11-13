#include "model/weight_map.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/logging.h"

namespace ography {

namespace {

float load_as_float(const void* data, DType dtype, size_t i) {
  switch (dtype) {
    case DType::F32: return static_cast<const float*>(data)[i];
    case DType::F16: return __half2float(static_cast<const __half*>(data)[i]);
    case DType::BF16: return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[i]);
    default: break;
  }
  throw std::invalid_argument("upload_tensor: unsupported source dtype " +
                              std::string(dtype_name(dtype)));
}

void store_from_float(void* data, DType dtype, size_t i, float v) {
  switch (dtype) {
    case DType::F32: static_cast<float*>(data)[i] = v; return;
    case DType::F16: static_cast<__half*>(data)[i] = __float2half(v); return;
    case DType::BF16: static_cast<__nv_bfloat16*>(data)[i] = __float2bfloat16(v); return;
    default: break;
  }
  throw std::invalid_argument("upload_tensor: unsupported destination dtype " +
                              std::string(dtype_name(dtype)));
}

void expect_shape(const Tensor& t, const std::vector<int64_t>& shape, const std::string& name) {
  if (t.shape() != shape) {
    std::string got, want;
    for (auto d : t.shape()) got += std::to_string(d) + ",";
    for (auto d : shape) want += std::to_string(d) + ",";
    throw std::runtime_error(name + ": shape [" + got + "] but config expects [" + want + "]");
  }
}

}  // namespace

Tensor upload_tensor(const Tensor& src, DType dst_dtype, bool transpose, DeviceArena& arena) {
  if (src.device() != Device::CPU) throw std::invalid_argument("upload_tensor: source must be CPU");
  if (transpose && src.ndim() != 2) {
    throw std::invalid_argument("upload_tensor: transpose needs a 2-d tensor");
  }

  std::vector<int64_t> dst_shape = src.shape();
  if (transpose) std::swap(dst_shape[0], dst_shape[1]);
  Tensor dst = arena.allocate(dst_dtype, dst_shape);

  // Fast path: same dtype, no transpose -> straight copy from the mmapped file.
  if (src.dtype() == dst_dtype && !transpose) {
    CUDA_CHECK(cudaMemcpy(dst.data(), src.data(), src.nbytes(), cudaMemcpyHostToDevice));
    return dst;
  }

  // Convert (and transpose) on the host into a staging buffer. Weight loading happens once
  // at startup, so simplicity beats speed here; a GPU-side convert can replace it later.
  std::vector<uint8_t> staging(dst.nbytes());
  const size_t n = static_cast<size_t>(src.numel());
  if (!transpose) {
    for (size_t i = 0; i < n; ++i) {
      store_from_float(staging.data(), dst_dtype, i, load_as_float(src.data(), src.dtype(), i));
    }
  } else {
    const size_t rows = static_cast<size_t>(src.dim(0));
    const size_t cols = static_cast<size_t>(src.dim(1));
    for (size_t r = 0; r < rows; ++r) {
      for (size_t c = 0; c < cols; ++c) {
        // src[r][c] -> dst[c][r]
        store_from_float(staging.data(), dst_dtype, c * rows + r,
                         load_as_float(src.data(), src.dtype(), r * cols + c));
      }
    }
  }
  CUDA_CHECK(cudaMemcpy(dst.data(), staging.data(), staging.size(), cudaMemcpyHostToDevice));
  return dst;
}

LlamaWeights map_llama_weights(const SafetensorsModel& checkpoint, const ModelConfig& config,
                               DeviceArena& arena) {
  OGRAPHY_TODO("map_llama_weights: name table, fuse qkv / gate_up, upload to GPU");
}

Gpt2Weights map_gpt2_weights(const SafetensorsModel& checkpoint, const ModelConfig& config,
                             DeviceArena& arena, DType compute_dtype) {
  const std::string prefix = checkpoint.contains("transformer.wte.weight") ? "transformer." : "";
  const int64_t H = config.hidden_size;
  const int64_t I = config.intermediate_size;

  auto vec = [&](const std::string& name, int64_t n) {
    Tensor t = checkpoint.get(prefix + name);
    expect_shape(t, {n}, name);
    return upload_tensor(t, compute_dtype, false, arena);
  };
  // Conv1D weight stored [in, out] -> engine layout [out, in].
  auto conv1d = [&](const std::string& name, int64_t in, int64_t out) {
    Tensor t = checkpoint.get(prefix + name);
    expect_shape(t, {in, out}, name);
    QuantizedWeight w;
    w.weight = upload_tensor(t, compute_dtype, /*transpose=*/true, arena);
    return w;
  };

  Gpt2Weights w;
  {
    Tensor wte = checkpoint.get(prefix + "wte.weight");
    expect_shape(wte, {config.vocab_size, H}, "wte.weight");
    w.wte = upload_tensor(wte, compute_dtype, false, arena);

    Tensor wpe = checkpoint.get(prefix + "wpe.weight");
    expect_shape(wpe, {config.max_position_embeddings, H}, "wpe.weight");
    w.wpe = upload_tensor(wpe, compute_dtype, false, arena);
  }

  w.layers.resize(config.num_layers);
  for (int i = 0; i < config.num_layers; ++i) {
    const std::string p = "h." + std::to_string(i) + ".";
    Gpt2LayerWeights& L = w.layers[i];
    L.ln_1_weight = vec(p + "ln_1.weight", H);
    L.ln_1_bias = vec(p + "ln_1.bias", H);
    L.wqkv = conv1d(p + "attn.c_attn.weight", H, 3 * H);
    L.bqkv = vec(p + "attn.c_attn.bias", 3 * H);
    L.wo = conv1d(p + "attn.c_proj.weight", H, H);
    L.bo = vec(p + "attn.c_proj.bias", H);
    L.ln_2_weight = vec(p + "ln_2.weight", H);
    L.ln_2_bias = vec(p + "ln_2.bias", H);
    L.w_fc = conv1d(p + "mlp.c_fc.weight", H, I);
    L.b_fc = vec(p + "mlp.c_fc.bias", I);
    L.w_proj = conv1d(p + "mlp.c_proj.weight", I, H);
    L.b_proj = vec(p + "mlp.c_proj.bias", H);
  }

  w.ln_f_weight = vec("ln_f.weight", H);
  w.ln_f_bias = vec("ln_f.bias", H);

  // GPT-2 ties lm_head to wte. An untied checkpoint would carry lm_head.weight ([vocab, H],
  // nn.Linear layout, no transpose needed).
  if (!config.tie_word_embeddings && checkpoint.contains("lm_head.weight")) {
    Tensor t = checkpoint.get("lm_head.weight");
    expect_shape(t, {config.vocab_size, H}, "lm_head.weight");
    w.lm_head.weight = upload_tensor(t, compute_dtype, false, arena);
  } else {
    w.lm_head.weight = w.wte;
  }
  return w;
}

}  // namespace ography
