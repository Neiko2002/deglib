cmake_minimum_required(VERSION 3.19)

project(benchmark)
include_directories(${PROJECT_SOURCE_DIR}/include)
set(DATA_PATH "" CACHE PATH "Path to data sets")

# build HNSW benchmark files
add_executable(hnswlib_anns_bench ${PROJECT_SOURCE_DIR}/src/hnswlib_anns_bench.cpp)
target_link_libraries(hnswlib_anns_bench PUBLIC HNSW_LIB compile-options
                                           fmt::fmt-header-only tsl::robin_map)
target_compile_definitions(hnswlib_anns_bench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

add_executable(hnswlib_explore_bench ${PROJECT_SOURCE_DIR}/src/hnswlib_explore_bench.cpp)
target_link_libraries(hnswlib_explore_bench PUBLIC HNSW_LIB compile-options
                                           fmt::fmt-header-only tsl::robin_map)
target_compile_definitions(hnswlib_explore_bench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

add_executable(hnswlib_graph_stats ${PROJECT_SOURCE_DIR}/src/hnswlib_graph_stats.cpp)
target_link_libraries(hnswlib_graph_stats PUBLIC HNSW_LIB compile-options
                                           fmt::fmt-header-only tsl::robin_map)
target_compile_definitions(hnswlib_graph_stats PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

# build DEG benchmark files
add_executable(deglib_anns_bench ${PROJECT_SOURCE_DIR}/src/deglib_anns_bench.cpp)
target_link_libraries(deglib_anns_bench PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only tsl::robin_map)
target_compile_definitions(deglib_anns_bench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

add_executable(deglib_explore_bench ${PROJECT_SOURCE_DIR}/src/deglib_explore_bench.cpp)
target_link_libraries(deglib_explore_bench PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only tsl::robin_map)
target_compile_definitions(deglib_explore_bench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

add_executable(deglib_build_bench ${PROJECT_SOURCE_DIR}/src/deglib_build_bench.cpp)
target_link_libraries(deglib_build_bench PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only tsl::robin_map)
target_compile_definitions(deglib_build_bench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

add_executable(deglib_graph_stats ${PROJECT_SOURCE_DIR}/src/deglib_graph_stats.cpp)
target_link_libraries(deglib_graph_stats PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only tsl::robin_map)
target_compile_definitions(deglib_graph_stats PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

add_executable(deglib_knng_bench ${PROJECT_SOURCE_DIR}/src/deglib_knng_bench.cpp)
target_link_libraries(deglib_knng_bench PUBLIC DEG_LIB compile-options
                                                fmt::fmt-header-only tsl::robin_map)
target_compile_definitions(deglib_knng_bench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")