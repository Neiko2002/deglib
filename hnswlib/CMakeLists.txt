cmake_minimum_required (VERSION 3.19)

project(hnswlib)

# setup the hnsw lib
# http://mariobadr.com/creating-a-header-only-library-with-cmake.html
add_library(HNSW_LIB INTERFACE)
target_include_directories(HNSW_LIB INTERFACE ${PROJECT_SOURCE_DIR}/include)

# https://www.brianlheim.com/2018/04/09/cmake-cheat-sheet.html