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

/**
 * priority queue with access to the internal data.
 * therefore access to the unsorted data is possible.
 * 
 * https://stackoverflow.com/questions/4484767/how-to-iterate-over-a-priority-queue
 * https://www.linuxtopia.org/online_books/programming_books/c++_practical_programming/c++_practical_programming_189.html
 */
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


}  // end namespace deglib


#include "memory.h"
#include "search.h"
#include "distances.h"
#include "repository.h"
#include "graph.h"
#include "readonly_graph.h"