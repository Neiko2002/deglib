cmake_minimum_required (VERSION 3.19)

project(examples)
include_directories(${PROJECT_SOURCE_DIR}/include)

# build executables
add_executable(update_test ${PROJECT_SOURCE_DIR}/src/updates_test.cpp)
target_link_libraries(update_test HNSW_LIB)

add_executable(searchKnnCloserFirst_test ${PROJECT_SOURCE_DIR}/src/searchKnnCloserFirst_test.cpp)
target_link_libraries(searchKnnCloserFirst_test HNSW_LIB)


add_executable(hnsw_index_tests ${PROJECT_SOURCE_DIR}/src/hnsw_index_tests.cpp)
target_link_libraries(hnsw_index_tests HNSW_LIB)
# https://cliutils.gitlab.io/modern-cmake/chapters/features/cpp11.html
#target_compile_features(hnsw_index_tests PUBLIC cxx_std_11)
#set_target_properties(hnsw_index_tests PROPERTIES CXX_EXTENSIONS OFF)