#include "model/weight_map.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <vector>

#include "core/cuda_check.h"
#include "core/logging.h"
#include "kernels/kernels.h"

namespace ography {

namespace {

template <typename Src, typename Dst>
void convert_typed(const void* src, void* dst, size_t n) {
  const Src* s = static_cast<const Src*>(src);
  Dst* d = static_cast<Dst*>(dst);
  for (size_t i = 0; i < n; ++i) {
    float v;
    if constexpr (std::is_same_v<Src, float>) v = s[i];
    else if constexpr (std::is_same_v<Src, __half>) v = __half2float(s[i]);
    else v = __bfloat162float(s[i]);
    if constexpr (std::is_same_v<Dst, float>) d[i] = v;
    else if constexpr (std::is_same_v<Dst, __half>) d[i] = __float2half(v);
    else d[i] = __float2bfloat16(v);
  }
}

// Converts n contiguous elements between F32/F16/BF16, dispatching once outside the loop.
void convert(const void* src, DType sd, void* dst, DType dd, size_t n) {
  auto with_dst = [&](auto src_tag) {
    using S = decltype(src_tag);
    switch (dd) {
      case DType::F32: return convert_typed<S, float>(src, dst, n);
      case DType::F16: return convert_typed<S, __half>(src, dst, n);
      case DType::BF16: return convert_typed<S, __nv_bfloat16>(src, dst, n);
      default:
        throw std::invalid_argument("upload_tensor: unsupported destination dtype " +
                                    std::string(dtype_name(dd)));
    }
  };
  switch (sd) {
    case DType::F32: return with_dst(float{});
    case DType::F16: return with_dst(__half{});
    case DType::BF16: return with_dst(__nv_bfloat16{});
    default:
      throw std::invalid_argument("upload_tensor: unsupported source dtype " +
                                  std::string(dtype_name(sd)));
  }
}

// Copies src (CPU) into device memory at dst, converting dtype if needed.
void upload_into(const Tensor& src, DType dst_dtype, void* dst) {
  if (src.dtype() == dst_dtype) {
    CUDA_CHECK(cudaMemcpy(dst, src.data(), src.nbytes(), cudaMemcpyHostToDevice));
    return;
  }
  const size_t n = static_cast<size_t>(src.numel());
  std::vector<uint8_t> staging(n * dtype_bits(dst_dtype) / 8);
  convert(src.data(), src.dtype(), staging.data(), dst_dtype, n);
  CUDA_CHECK(cudaMemcpy(dst, staging.data(), staging.size(), cudaMemcpyHostToDevice));
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

  if (!transpose) {
    upload_into(src, dst_dtype, dst.data());
    return dst;
  }

  // Transpose on the host (via fp32), then upload. Weight loading happens once at startup,
  // so simplicity beats speed here.
  const size_t rows = static_cast<size_t>(src.dim(0));
  const size_t cols = static_cast<size_t>(src.dim(1));
  std::vector<float> row(cols), transposed(rows * cols);
  for (size_t r = 0; r < rows; ++r) {
    convert(static_cast<const uint8_t*>(src.data()) + r * cols * dtype_bits(src.dtype()) / 8,
            src.dtype(), row.data(), DType::F32, cols);
    for (size_t c = 0; c < cols; ++c) transposed[c * rows + r] = row[c];  // src[r][c] -> dst[c][r]
  }
  Tensor staged(transposed.data(), DType::F32, dst_shape, Device::CPU);
  upload_into(staged, dst_dtype, dst.data());
  return dst;
}

Tensor upload_concat(const std::vector<Tensor>& parts, DType dst_dtype, DeviceArena& arena) {
  if (parts.empty()) throw std::invalid_argument("upload_concat: no parts");
  std::vector<int64_t> shape = parts[0].shape();
  shape[0] = 0;
  for (const Tensor& p : parts) {
    std::vector<int64_t> rest = p.shape();
    rest[0] = 0;
    if (p.device() != Device::CPU || rest != shape) {
      throw std::invalid_argument("upload_concat: parts must be CPU tensors with matching trailing dims");
    }
  }
  for (const Tensor& p : parts) shape[0] += p.dim(0);
  Tensor dst = arena.allocate(dst_dtype, shape);
  size_t offset = 0;
  for (const Tensor& p : parts) {
    upload_into(p, dst_dtype, static_cast<uint8_t*>(dst.data()) + offset);
    offset += static_cast<size_t>(p.numel()) * dtype_bits(dst_dtype) / 8;
  }
  return dst;
}

namespace {

// A CPU tensor copied to a temporary device allocation (freed on scope exit). Only used while
// loading, so plain cudaMalloc is fine.
struct TempDevice {
  Tensor t;
  explicit TempDevice(const Tensor& src) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, src.nbytes()));
    CUDA_CHECK(cudaMemcpy(p, src.data(), src.nbytes(), cudaMemcpyHostToDevice));
    t = Tensor(p, src.dtype(), src.shape(), Device::CUDA);
  }
  ~TempDevice() { cudaFree(t.data()); }
  TempDevice(const TempDevice&) = delete;
  TempDevice& operator=(const TempDevice&) = delete;
};

// Loads checkpoint linears (each "<prefix>.qweight/.qzeros/.scales", [in, out_i]) fused along
// out into one engine-layout INT4 weight.
QuantizedWeight load_int4(const SafetensorsModel& ckpt, const QuantConfig& q,
                          const std::vector<std::pair<std::string, int64_t>>& parts, int64_t in,
                          DeviceArena& arena) {
  int64_t total_out = 0;
  for (const auto& [name, out] : parts) total_out += out;
  const int64_t groups = in / q.group_size;
  QuantizedWeight w;
  w.method = q.method;
  w.group_size = q.group_size;
  w.weight = arena.allocate(DType::I32, {total_out, in / 8});
  w.scales = arena.allocate(DType::F16, {total_out, groups});
  w.zeros = arena.allocate(DType::F16, {total_out, groups});
  const auto source =
      q.method == QuantMethod::AWQ ? kernels::Int4Source::AWQ : kernels::Int4Source::GPTQ;

  int64_t row = 0;
  for (const auto& [name, out] : parts) {
    const Tensor qweight = ckpt.get(name + ".qweight");
    const Tensor qzeros = ckpt.get(name + ".qzeros");
    const Tensor scales = ckpt.get(name + ".scales");
    if (source == kernels::Int4Source::AWQ) expect_shape(qweight, {in, out / 8}, name + ".qweight");
    else expect_shape(qweight, {in / 8, out}, name + ".qweight");
    expect_shape(qzeros, {groups, out / 8}, name + ".qzeros");
    expect_shape(scales, {groups, out}, name + ".scales");
    if (scales.dtype() != DType::F16) throw std::runtime_error(name + ".scales must be f16");
    if (q.method == QuantMethod::GPTQ && ckpt.contains(name + ".g_idx")) {
      // Act-order is rejected by QuantConfig, but check the permutation really is trivial.
      const Tensor g_idx = ckpt.get(name + ".g_idx");
      const int32_t* g = g_idx.data_as<const int32_t>();
      for (int64_t i = 0; i < g_idx.numel(); ++i) {
        if (g[i] != i / q.group_size) throw std::runtime_error(name + ": g_idx is not sequential");
      }
    }
    TempDevice d_qweight(qweight), d_qzeros(qzeros), d_scales(scales);
    kernels::repack_int4(source, d_qweight.t, d_qzeros.t, d_scales.t, q.group_size, w.weight,
                         w.scales, w.zeros, static_cast<int>(row), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());  // before the temporaries are freed
    row += out;
  }
  return w;
}

}  // namespace

LlamaWeights map_llama_weights(const SafetensorsModel& checkpoint, const ModelConfig& config,
                               DeviceArena& arena, DType compute_dtype) {
  const int64_t H = config.hidden_size;
  const int64_t I = config.intermediate_size;
  const int64_t q_out = int64_t(config.num_heads) * config.head_dim;
  const int64_t kv_out = int64_t(config.num_kv_heads) * config.head_dim;

  auto get = [&](const std::string& name, std::vector<int64_t> shape) {
    Tensor t = checkpoint.get(name);
    expect_shape(t, shape, name);
    return t;
  };
  auto vec = [&](const std::string& name, int64_t n) {
    return upload_tensor(get(name, {n}), compute_dtype, false, arena);
  };
  auto linear = [&](const std::string& name, int64_t out, int64_t in) {
    QuantizedWeight w;
    w.weight = upload_tensor(get(name, {out, in}), compute_dtype, false, arena);
    return w;
  };
  const bool int4 = config.quant.method == QuantMethod::AWQ || config.quant.method == QuantMethod::GPTQ;
  // A decoder linear: names are module prefixes ("...q_proj"), fused along out when several.
  auto proj = [&](const std::vector<std::pair<std::string, int64_t>>& parts, int64_t in) {
    if (int4) return load_int4(checkpoint, config.quant, parts, in, arena);
    std::vector<Tensor> tensors;
    for (const auto& [name, out] : parts) tensors.push_back(get(name + ".weight", {out, in}));
    QuantizedWeight w;
    w.weight = upload_concat(tensors, compute_dtype, arena);
    return w;
  };

  LlamaWeights w;
  w.embed_tokens = upload_tensor(get("model.embed_tokens.weight", {config.vocab_size, H}),
                                 compute_dtype, false, arena);
  w.layers.resize(config.num_layers);
  for (int i = 0; i < config.num_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    LlamaLayerWeights& L = w.layers[i];
    L.attn_norm = vec(p + "input_layernorm.weight", H);
    L.wqkv = proj({{p + "self_attn.q_proj", q_out},
                   {p + "self_attn.k_proj", kv_out},
                   {p + "self_attn.v_proj", kv_out}},
                  H);
    if (config.attention_bias) {
      L.bqkv = upload_concat({get(p + "self_attn.q_proj.bias", {q_out}),
                              get(p + "self_attn.k_proj.bias", {kv_out}),
                              get(p + "self_attn.v_proj.bias", {kv_out})},
                             compute_dtype, arena);
    }
    L.wo = proj({{p + "self_attn.o_proj", H}}, q_out);
    L.ffn_norm = vec(p + "post_attention_layernorm.weight", H);
    L.w_gate_up = proj({{p + "mlp.gate_proj", I}, {p + "mlp.up_proj", I}}, H);
    L.w_down = proj({{p + "mlp.down_proj", H}}, I);
  }
  w.final_norm = vec("model.norm.weight", H);

  if (!config.tie_word_embeddings && checkpoint.contains("lm_head.weight")) {
    w.lm_head = linear("lm_head.weight", config.vocab_size, H);
  } else {
    w.lm_head.weight = w.embed_tokens;
  }
  return w;
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
