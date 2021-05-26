#pragma once

#ifndef NO_MANUAL_VECTORIZATION
  #ifdef _MSC_VER
    #if (defined(_M_AMD64) || defined(_M_X64) || defined(_M_IX86_FP) == 2)
      #define __SSE__
      #define __SSE2__
    #elif defined(_M_IX86_FP) == 1
      #define __SSE__
    #endif
  #endif

  #ifdef __SSE__
    #define USE_SSE
    #ifdef __AVX__
      #define USE_AVX
    #endif
  #endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
  #ifdef _MSC_VER
    #include <intrin.h>
    #include <stdexcept>
  #else
    #include <x86intrin.h>
  #endif

  #if defined(__GNUC__)
    #define PORTABLE_ALIGN32 __attribute__((aligned(32)))
    #define PORTABLE_ALIGN16 __attribute__((aligned(16)))
  #else
    #define PORTABLE_ALIGN32 __declspec(align(32))
    #define PORTABLE_ALIGN16 __declspec(align(16))
  #endif
#endif

#ifdef _WINDOWS
  #include <malloc.h>
  #define vla(var_name, dtype, size) auto var_name = (dtype*) _malloca(size*sizeof(dtype));
  #define free_vla(arr) _freea(arr);
#else
  #define vla(var_name, dtype, size) dtype var_name[size];
  #define free_vla(arr) 
#endif



#include "memory.h"
#include "distances.h"
#include "search.h"
#include "repository.h"
#include "graph.h"
#include "graph/readonly_graph.h"
#include "graph/sizebounded_graph.h"
#include "builder.h"