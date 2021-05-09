#pragma once



namespace deglib
{

    template <typename MTYPE>
    using DISTFUNC = MTYPE (*)(const void *, const void *, const void *);

    template <typename MTYPE>
    class SpaceInterface
    {
    public:

        virtual const size_t get_data_size() const = 0;

        virtual const DISTFUNC<MTYPE> get_dist_func() const = 0;

        virtual const void *get_dist_func_param() const = 0;
    };

    int test_func() {
        fmt::print("deglib test");
        return 10;
    }

}  // end namespace deglib


#include "repository.h"
#include "graph.h"
#include "distances.h"
#include "search.h"