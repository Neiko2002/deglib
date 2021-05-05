cmake_minimum_required (VERSION 3.19)

# Only interpret if() arguments as variables or keywords when unquoted.
# More details run "cmake --help-policy CMP0054"
cmake_policy(SET CMP0054 NEW)

project(deg_lib)
set(PROJECT_URL "https://github.com/Neiko2002/DEG")
set(PROJECT_DESCRIPTION " Dynamic Exploration Graphs for fast approximate nearest neighbor search and navigation in large image datasets")
# Need to bump this when we release a new version.
set(PROJECT_VERSION "0.0.1")

# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

# detecting CPU instruction support
# https://github.com/lemire/FastPFor/blob/master/CMakeLists.txt
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules")
include(DetectCPUFeatures)

# detecting openmp
find_package(OpenMP)
if(OPENMP_FOUND)
  message("Found OpenMP")
endif()

# setup compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -std=c++17 -DHAVE_CXX0X -openmp -march=native -fpic -ftree-vectorize")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++17 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0")
elseif (MSVC)
  # https://stackoverflow.com/questions/54114287/visual-studio-not-recognizing-avx2-or-avx/54137338#54137338  
  set(CMAKE_CXX_FLAGS "/fp:fast /GS- ${CMAKE_CXX_FLAGS}")

  # detecting SUPPORT_AVX2 or SSE2 support
  cmake_host_system_information(RESULT SUPPORT_SSE2 QUERY HAS_SSE2)
  if(SUPPORT_AVX2)
    add_compile_definitions(_AVX2__=1)
    set(CMAKE_CXX_FLAGS "/arch:AVX2 ${CMAKE_CXX_FLAGS}")
  elseif(SUPPORT_SSE2)
    set(CMAKE_CXX_FLAGS "/arch:SSE2 ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "/arch:native ${CMAKE_CXX_FLAGS}")
  endif()

  # use openmp if possible
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "/openmp ${CMAKE_CXX_FLAGS}")
  endif()

  # disable string optimizations and function level linking
  # https://stackoverflow.com/questions/5063334/what-is-the-difference-between-the-ox-and-o2-compiler-options
  string(REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})

  # old settings
  #set( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize" )
endif()


message("C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message("C++ compiler flags release: ${CMAKE_CXX_FLAGS_RELEASE}")

add_subdirectory(external/fmt) 
add_subdirectory(external/robin-map) 

add_subdirectory(hnswlib) 
add_subdirectory(deglib) 
add_subdirectory(benchmark) 
add_subdirectory(examples)
