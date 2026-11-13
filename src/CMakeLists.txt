# One static library per engine module. Dependency direction (no cycles):
#
#   core <- kernels <- kv_cache <- model <- engine <- http (src/server)
#   core <- quant   <- model
#   tokenizer (CPU only) <- engine
#
# Every target exposes src/ as its include root, so includes look like "core/tensor.h".

function(ography_module name)
  add_library(ography_${name} STATIC ${ARGN})
  target_include_directories(ography_${name} PUBLIC ${PROJECT_SOURCE_DIR}/src)
  set_target_properties(ography_${name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
  ography_set_flags(ography_${name})
endfunction()

ography_module(core
  core/dtype.cpp
  core/tensor.cpp
  core/memory.cpp
  core/logging.cpp)
target_link_libraries(ography_core PUBLIC CUDA::cudart)

ography_module(kernels
  kernels/embedding.cu
  kernels/rmsnorm.cu
  kernels/layernorm.cu
  kernels/gelu.cu
  kernels/qkv.cu
  kernels/rope.cu
  kernels/swiglu.cu
  kernels/attention_prefill.cu
  kernels/attention_decode.cu
  kernels/gemm.cu
  kernels/sampling.cu
  kernels/dequant.cu)
target_link_libraries(ography_kernels PUBLIC ography_core CUDA::cublas)

ography_module(kv_cache
  kv_cache/kv_cache.cpp
  kv_cache/prefix_reuse.cpp)
target_link_libraries(ography_kv_cache PUBLIC ography_core)

ography_module(quant
  quant/quant_config.cpp)
target_link_libraries(ography_quant PUBLIC ography_core nlohmann_json::nlohmann_json)

ography_module(model
  model/config.cpp
  model/safetensors.cpp
  model/weight_map.cpp
  model/registry.cpp
  model/arch/llama.cpp
  model/arch/gpt2.cpp)
target_link_libraries(ography_model PUBLIC
  ography_core ography_kernels ography_kv_cache ography_quant nlohmann_json::nlohmann_json)

ography_module(tokenizer
  tokenizer/tokenizer.cpp
  tokenizer/bpe.cpp
  tokenizer/pretokenizer.cpp
  tokenizer/unicode.cpp
  tokenizer/chat_template.cpp)
target_link_libraries(ography_tokenizer PUBLIC nlohmann_json::nlohmann_json)

ography_module(engine
  engine/engine.cpp
  engine/stop_checker.cpp
  engine/cuda_graph.cpp
  engine/request_queue.cpp)
target_link_libraries(ography_engine PUBLIC ography_model ography_tokenizer ography_kv_cache)

ography_module(http
  server/http_server.cpp
  server/openai_api.cpp
  server/sse.cpp)
target_link_libraries(ography_http PUBLIC
  ography_engine httplib::httplib nlohmann_json::nlohmann_json)
