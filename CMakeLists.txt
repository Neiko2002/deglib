cmake_minimum_required (VERSION 3.1)

# Only interpret if() arguments as variables or keywords when unquoted.
# More details run "cmake --help-policy CMP0054"
cmake_policy(SET CMP0054 NEW)



project (deg_lib)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  SET( CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -ftree-vectorize")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize" )
endif()



# setup the hnsw lib
# http://mariobadr.com/creating-a-header-only-library-with-cmake.html
add_library(HNSW_LIB INTERFACE)
target_include_directories(HNSW_LIB INTERFACE "src/hnswlib/")

# build executables 
add_executable(sift_1m "src/benchmark/sift_1m.cpp")
target_link_libraries(sift_1m HNSW_LIB)

add_executable(sift_1b "src/benchmark/sift_1b.cpp")
target_link_libraries(sift_1b HNSW_LIB)

add_executable(sift_test "src/benchmark/sift_test.cpp")
target_link_libraries(sift_test HNSW_LIB)

add_executable(update_test "src/examples/updates_test.cpp")
target_link_libraries(update_test HNSW_LIB)

add_executable(searchKnnCloserFirst_test "src/examples/searchKnnCloserFirst_test.cpp")
target_link_libraries(searchKnnCloserFirst_test HNSW_LIB)