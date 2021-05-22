cmake_minimum_required(VERSION 3.19)

project(benchmark)
include_directories(${PROJECT_SOURCE_DIR}/include)

# build executables
add_executable(sift_1m ${PROJECT_SOURCE_DIR}/src/sift_1m.cpp)
target_link_libraries(sift_1m HNSW_LIB)

add_executable(sift_1b ${PROJECT_SOURCE_DIR}/src/sift_1b.cpp)
target_link_libraries(sift_1b HNSW_LIB)

add_executable(sift_test ${PROJECT_SOURCE_DIR}/src/sift_test.cpp)
target_link_libraries(sift_test HNSW_LIB)

set(DATA_PATH "" CACHE PATH "Path to data")
add_executable(hnswlib_bench ${PROJECT_SOURCE_DIR}/src/hnswlib_bench.cpp)
target_link_libraries(hnswlib_bench PUBLIC HNSW_LIB compile-options
                                           fmt::fmt-header-only tsl::robin_map)
target_compile_definitions(hnswlib_bench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

# build DEG tests files
add_executable(readonly_graph_test ${PROJECT_SOURCE_DIR}/src/deglib_readonly_graph_test.cpp)
target_link_libraries(readonly_graph_test PUBLIC DEG_LIB compile-options
                                      fmt::fmt-header-only tsl::robin_map cntgs::cntgs)
target_compile_definitions(readonly_graph_test PUBLIC "DATA_PATH=\"${DATA_PATH}\"")

add_executable(deglib_bench ${PROJECT_SOURCE_DIR}/src/deglib_bench.cpp)
target_link_libraries(deglib_bench PUBLIC DEG_LIB compile-options
                                          fmt::fmt-header-only tsl::robin_map)
target_compile_definitions(deglib_bench PUBLIC "DATA_PATH=\"${DATA_PATH}\"")
