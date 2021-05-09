cmake_minimum_required (VERSION 3.19)

project(benchmark)
include_directories(${PROJECT_SOURCE_DIR}/include)

# build executables 
add_executable(sift_1m ${PROJECT_SOURCE_DIR}/src/sift_1m.cpp)
target_link_libraries(sift_1m HNSW_LIB)

add_executable(sift_1b ${PROJECT_SOURCE_DIR}/src/sift_1b.cpp)
target_link_libraries(sift_1b HNSW_LIB)

add_executable(sift_test ${PROJECT_SOURCE_DIR}/src/sift_test.cpp)
target_link_libraries(sift_test HNSW_LIB)


add_executable(hnswlib_bench ${PROJECT_SOURCE_DIR}/src/hnswlib_bench.cpp)
target_link_libraries(hnswlib_bench HNSW_LIB)
target_link_libraries(hnswlib_bench fmt::fmt-header-only)
target_link_libraries(hnswlib_bench tsl::robin_map)



# build DEG tests files
add_executable(deg_test ${PROJECT_SOURCE_DIR}/src/deg_test.cpp)
target_compile_features(deg_test PUBLIC cxx_std_17)
target_link_libraries(deg_test DEG_LIB)
target_link_libraries(deg_test fmt::fmt-header-only)
target_link_libraries(deg_test tsl::robin_map)


add_executable(deglib_bench ${PROJECT_SOURCE_DIR}/src/deglib_bench.cpp)
target_compile_features(deglib_bench PUBLIC cxx_std_17)
target_link_libraries(deglib_bench DEG_LIB)
target_link_libraries(deglib_bench fmt::fmt-header-only)
target_link_libraries(deglib_bench tsl::robin_map)