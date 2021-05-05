
#include "hnswlib.h"

#include <iostream>
#include <omp.h>

int main()
{
    std::cout << "Start ..." << std::endl;

#pragma omp parallel for
    for (int i = 1; i < 10; i++)
    {
        int threadID = omp_get_thread_num();
#pragma omp critical
        {
            std::printf("Thread %d reporting\n", threadID);
        }
    }


    #ifdef _AVX2__
        std::printf(" cmake AVX2\n");
    #endif

    #ifdef __AVX2__
        std::printf(" use AVX2\n");
    #endif

    #ifndef __AVX2__
        std::printf(" no AVX2\n");
    #endif

    #ifdef __SSE__
        std::printf(" use SSE\n");
    #endif

    #ifndef __SSE__
        std::printf(" no SSE\n");
    #endif

    #ifdef USE_SSE
        std::printf(" use SSE\n");
    #endif

    #ifndef USE_SSE
        std::printf("no SSE\n");
    #endif



    size_t vecdim = 128;
    char *path_index = "c:/Data/Feature/SIFT1M/sift1m_ef_500_M_16.bin";

    hnswlib::L2Space l2space(vecdim);
    hnswlib::HierarchicalNSW<float> *appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, path_index, false);

    //std::cout << appr_alg->element_levels_ << std::endl;

    std::cout << "... Finished" << std::endl;

    return 0;
}
