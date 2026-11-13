# Executables. Library targets are named ography_<module>; the src/server module's
# target is ography_http so it doesn't clash with the ography_server binary.

add_executable(ography_server ography_server.cpp)
target_link_libraries(ography_server PRIVATE ography_http)
ography_set_flags(ography_server)

add_executable(ography_cli ography_cli.cpp)
target_link_libraries(ography_cli PRIVATE ography_engine)
ography_set_flags(ography_cli)

add_executable(ography_bench ography_bench.cpp)
target_link_libraries(ography_bench PRIVATE ography_engine)
ography_set_flags(ography_bench)
