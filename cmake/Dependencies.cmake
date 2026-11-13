# Third-party dependencies, fetched at configure time (sources land in the build dir,
# not third_party/; see third_party/README.md).
include(FetchContent)

FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE)

FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG        v0.18.1
  GIT_SHALLOW    TRUE)

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE INTERNAL "")
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json httplib)

if(OGRAPHY_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE)
  set(INSTALL_GTEST OFF CACHE INTERNAL "")
  FetchContent_MakeAvailable(googletest)
endif()
