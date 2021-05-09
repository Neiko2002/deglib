
// This is a test file for testing the interface
//  >>> virtual std::vector<std::pair<dist_t, labeltype>>
//  >>>    searchKnnCloserFirst(const void* query_data, size_t k) const;
// of class AlgorithmInterface

#include "hnswlib.h"
#include "utils.h"

#include <vector>
#include <iostream>
#include <omp.h>

/**
 * Add memmove and read all base features at once
 * https://github.com/facebookresearch/faiss/blob/master/demos/demo_sift1M.cpp
 */
void sift_1m()
{

    size_t query_size = 10000;
    size_t base_size = 1000000;
    size_t top_k = 100;
    size_t vecdim = 128;

    int efConstruction = 500;
    int M = 16;

    char *path_query = "c:/Data/Feature/SIFT1M/sift_query.fvecs";
    char *path_groundtruth = "c:/Data/Feature/SIFT1M/sift_groundtruth.ivecs";
    char *path_basedata = "c:/Data/Feature/SIFT1M/sift_base.fvecs";
    char path_index[1024];
    sprintf(path_index, "c:/Data/Feature/SIFT1M/sift1m_ef_%d_M_%d.bin", efConstruction, M);

    float *feature = new float[vecdim];

    std::cout << "Loading ground truth data:\n";
    unsigned int *groundtruth = new unsigned int[query_size * top_k];
    {
        std::ifstream inputGT(path_groundtruth, std::ios::binary);
        for (int i = 0; i < query_size; i++)
        {
            int len;
            inputGT.read((char *)&len, sizeof(int));
            if (len != top_k)
            {
                std::cout << "err";
                return;
            }
            inputGT.read((char *)(groundtruth + top_k * i), len * sizeof(int));
        }
        inputGT.close();
    }

    std::cout << "Loading query data:" << std::endl;
    float *query_features = new float[query_size * vecdim];
    {
        std::ifstream inputQ(path_query, std::ios::binary);
        for (int i = 0; i < query_size; i++)
        {

            // check vector dims with the information in the file
            int len;
            inputQ.read((char *)&len, sizeof(int));
            if (len != vecdim)
            {
                std::cout << "file error";
                exit(1);
            }
            inputQ.read((char *)(query_features + vecdim * i), len * sizeof(float));
        }
        inputQ.close();
    }

    std::cout << "Loading base data:" << std::endl;
    float *base_features = new float[base_size * vecdim];
    {
        std::ifstream inputB(path_basedata, std::ios::binary);
        for (int i = 0; i < base_size; i++)
        {
            int len;
            inputB.read((char *)&len, sizeof(int));
            if (len != vecdim)
            {
                std::cout << "file error";
                exit(1);
            }
            inputB.read((char *)(base_features + vecdim * i), len * sizeof(float));
        }
        inputB.close();
    }

    hnswlib::L2Space l2space(vecdim);
    hnswlib::HierarchicalNSW<float> *appr_alg;
    if (exists_test(path_index))
    {
        std::cout << "Loading index from " << path_index << ":" << std::endl;
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, path_index, false);
        std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb" << std::endl;
    }
    else
    {
        std::cout << "Building index:" << std::endl;
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, base_size, M, efConstruction);

        int j1 = 0;
        appr_alg->addPoint((void *)(base_features), (size_t)j1);

        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 10000;

        omp_set_num_threads(1);
#pragma omp parallel for
        for (int i = 1; i < base_size; i++)
        {
            int j2 = 0;

#pragma omp critical
            {
                j1++;
                j2 = j1;
                if (j1 % report_every == 0)
                {
                    std::cout << j1 / (0.01 * base_size) << " %, "
                              << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips "
                              << " Mem: "
                              << getCurrentRSS() / 1000000 << " Mb" << std::endl;
                    stopw.reset();
                }
            }
            appr_alg->addPoint((void *)(base_features + vecdim * j1), (size_t)j2);
        }

        std::cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds" << std::endl;
        appr_alg->saveIndex(path_index);
    }

    vector<std::priority_queue<std::pair<float, labeltype>>> answers;
    size_t k = 10; // k at test time
    std::cout << "Parsing gt:" << std::endl;
    get_gt(groundtruth, query_features, base_features, base_size, query_size, l2space, vecdim, answers, k);
    std::cout << "Loaded gt" << std::endl;
    for (int i = 0; i < 1; i++)
        test_vs_recall(query_features, base_size, query_size, *appr_alg, vecdim, answers, k);
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb" << std::endl;

    return;
}

int main()
{
    std::cout << "Testing ..." << std::endl;
    sift_1m();
    std::cout << "Test ok" << std::endl;

    return 0;
}