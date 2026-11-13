# third_party

Dependencies are fetched by `cmake/Dependencies.cmake` (FetchContent) into the build
directory, so nothing is vendored here yet. Put any vendored sources here if a dependency
ever needs patching or offline builds are required.

- nlohmann/json v3.11.3: JSON for config, tokenizer and API bodies
- cpp-httplib v0.18.1: HTTP server
- GoogleTest v1.15.2: tests
