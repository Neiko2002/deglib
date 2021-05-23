cmake_minimum_required (VERSION 3.19)

project(deglib)
#include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(cntgs)

# setup the deg lib
# http://mariobadr.com/creating-a-header-only-library-with-cmake.html
add_library(DEG_LIB INTERFACE)
target_link_libraries(DEG_LIB INTERFACE tsl::robin_map)
target_link_libraries(DEG_LIB INTERFACE fmt::fmt-header-only cntgs::cntgs)
target_include_directories(DEG_LIB INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(DEG_LIB INTERFACE cxx_std_20)
