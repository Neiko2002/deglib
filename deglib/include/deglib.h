#pragma once

#ifndef NO_MANUAL_VECTORIZATION
#ifdef _MSC_VER
#if (defined(_M_AMD64) || defined(_M_X64) || _M_IX86_FP == 2)
#define __SSE__
#define __SSE2__
#elif _M_IX86_FP == 1
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
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

#include <fmt/core.h>

namespace deglib
{
template <typename MTYPE>
using DISTFUNC = MTYPE (*)(const void*, const void*, const void*);

template <typename MTYPE>
class SpaceInterface
{
  public:
    virtual const size_t get_data_size() const = 0;

    virtual const DISTFUNC<MTYPE> get_dist_func() const = 0;

    virtual const void* get_dist_func_param() const = 0;
};

int test_func()
{
    fmt::print("deglib test");
    return 10;
}

}  // end namespace deglib

#include "search.h"
#include "distances.h"
#include "repository.h"
#include "graph.h"