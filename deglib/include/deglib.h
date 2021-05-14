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

#include <queue>
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


//#pragma pack(2)

class ObjectDistance
{
    uint32_t id_;
    float distance_;

  public:
    ObjectDistance(const uint32_t id, const float distance) : id_(id), distance_(distance) {}

    inline const uint32_t getId() const { return id_; }

    inline const float getDistance() const { return distance_; }

    inline bool operator==(const ObjectDistance& o) const { return (distance_ == o.distance_) && (id_ == o.id_); }

    inline bool operator<(const ObjectDistance& o) const
    {
        if (distance_ == o.distance_)
        {
            return id_ < o.id_;
        }
        else
        {
            return distance_ < o.distance_;
        }
    }

    inline bool operator>(const ObjectDistance& o) const
    {
        if (distance_ == o.distance_)
        {
            return id_ > o.id_;
        }
        else
        {
            return distance_ > o.distance_;
        }
    }
};

//#pragma pack()


template<class T, class Compare>
class PQV : public std::vector<T> {
  Compare comp;
  public:
    PQV(Compare cmp = Compare()) : comp(cmp) {
      std::make_heap(this->begin(),this->end(), comp);
    }

    const T& top() { return this->front(); }

    void push(const T& x) {
      this->push_back(x);
      std::push_heap(this->begin(),this->end(), comp);
    }

    void pop() {
      std::pop_heap(this->begin(),this->end(), comp);
      this->pop_back();
    }
};

// search result set containing node ids and distances
typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance>> ResultSet;

// set of unchecked node ids
typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::greater<ObjectDistance>> UncheckedSet;


typedef PQV<ObjectDistance, std::less<ObjectDistance>> ResultQueue;


class SearchGraph
{
  public:    
    virtual deglib::ResultSet yahooSearch(const std::vector<uint32_t>& entry_node_indizies, const float* query, const float eps, const int k)  const = 0;
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
#include "readonly_graph.h"