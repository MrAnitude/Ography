include(GoogleTest)

# Unit tests: pure C++ logic (no GPU needed for most).
add_executable(unit_tests
  unit/test_dtype.cpp
  unit/test_tensor.cpp
  unit/test_safetensors.cpp
  unit/test_config.cpp
  unit/test_tokenizer.cpp
  unit/test_stop_checker.cpp
  unit/test_openai_api.cpp
  unit/test_rope.cpp)
target_link_libraries(unit_tests PRIVATE ography_http GTest::gtest_main)
target_compile_definitions(unit_tests PRIVATE OGRAPHY_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)

# Kernel tests: each kernel against a straightforward CPU reference. Needs the GPU.
add_executable(kernel_tests
  kernels/test_rmsnorm.cpp
  kernels/test_elementwise.cpp
  kernels/test_attention.cpp
  kernels/test_sampling.cpp
  kernels/test_int4.cpp)
target_link_libraries(kernel_tests PRIVATE ography_kernels GTest::gtest_main)
gtest_discover_tests(kernel_tests DISCOVERY_MODE PRE_TEST)

# Model tests: device arena and real-checkpoint weight loading. Needs the GPU; the GPT-2
# tests skip themselves if models/gpt2 (or $OGRAPHY_GPT2_DIR) is absent.
add_executable(model_tests
  model/test_arena.cpp
  model/test_gpt2_load.cpp
  model/test_gpt2_forward.cpp
  model/test_engine.cpp
  model/test_server.cpp)
target_link_libraries(model_tests PRIVATE ography_http GTest::gtest_main)
target_compile_definitions(model_tests PRIVATE OGRAPHY_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
gtest_discover_tests(model_tests DISCOVERY_MODE PRE_TEST)

# Golden tests: compare layer outputs / logits with reference tensors dumped from
# HF transformers by tools/dump_golden.py. Uses $OGRAPHY_GOLDEN_DIR, else golden/gpt2;
# skipped if neither exists.
add_executable(golden_tests golden/test_golden.cpp)
target_link_libraries(golden_tests PRIVATE ography_engine GTest::gtest_main)
target_compile_definitions(golden_tests PRIVATE OGRAPHY_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
gtest_discover_tests(golden_tests DISCOVERY_MODE PRE_TEST)
# The same tests against the Llama-family references (skipped until dumped).
foreach(model llama3.2-1b qwen2.5-1.5b qwen2.5-1.5b-awq qwen2.5-1.5b-gptq)
  add_test(NAME golden_${model} COMMAND golden_tests)
  set_tests_properties(golden_${model} PROPERTIES
    ENVIRONMENT "OGRAPHY_GOLDEN_DIR=${PROJECT_SOURCE_DIR}/golden/${model}"
    SKIP_REGULAR_EXPRESSION "NO GOLDEN DATA")
endforeach()
