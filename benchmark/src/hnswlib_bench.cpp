#include <omp.h>
#include <fmt/core.h>
#include <tsl/robin_set.h>

#include <algorithm>

#include "hnswlib.h"
#include "hnsw/utils.h"
#include "stopwatch.h"

static std::vector<tsl::robin_set<size_t>> get_ground_truth(const uint32_t* ground_truth,
                                                            const size_t ground_truth_size, const size_t k)
{
    auto answers = std::vector<tsl::robin_set<size_t>>(ground_truth_size);
    answers.reserve(ground_truth_size);
    for (int i = 0; i < ground_truth_size; i++)
    {
        auto& gt = answers[i];
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) gt.insert(ground_truth[k * i + j]);
    }

    return answers;
}

static float test_approx(const hnswlib::HierarchicalNSW<float>& appr_alg, const float* query_repository,
                         const std::vector<tsl::robin_set<size_t>>& ground_truth, const size_t query_dims,
                         const size_t k)
{
    size_t correct = 0;
    size_t total = 0;
    for (int i = 0; i < ground_truth.size(); i++)
    {
        auto gt = ground_truth[i];
        auto query = query_repository + query_dims * i;
        auto result_queue = appr_alg.searchKnn(query, k);

        total += gt.size();
        while (result_queue.empty() == false)
        {
            if (gt.find(result_queue.top().second) != gt.end()) correct++;
            result_queue.pop();
        }
    }

    return 1.0f * correct / total;
}

static void test_vs_recall(hnswlib::HierarchicalNSW<float>& appr_alg, const float* query_repository,
                           const std::vector<tsl::robin_set<size_t>>& ground_truth, const size_t query_dims,
                           const size_t k)
{
    std::vector<size_t> efs = { 100, 121, 145, 180 };
    /*std::vector<size_t> efs;  // = { 10,10,10,10,10 };
    for (int i = k; i < 30; i++)
    {
        efs.push_back(i);
    }
    for (int i = std::max<size_t>(30, k); i < 100; i += 10)
    {
        efs.push_back(i);
    }
    for (int i = std::max<size_t>(100, k); i < 500; i += 40)
    {
        efs.push_back(i);
    }*/
    for (size_t ef : efs)
    {
        appr_alg.setEf(ef);
        appr_alg.metric_distance_computations = 0;
        appr_alg.metric_hops = 0;

        auto stopw = StopW();
        auto recall = test_approx(appr_alg, query_repository, ground_truth, query_dims, k);
        auto time_us_per_query = stopw.getElapsedTimeMicro() / ground_truth.size();

        float distance_comp_per_query = appr_alg.metric_distance_computations / (1.0f * ground_truth.size());
        float hops_per_query = appr_alg.metric_hops / (1.0f * ground_truth.size());

        fmt::print("ef {} \t recall {} \t time_us_per_query {}us, avg distance computations {}, avg hops {}\n", ef,
                   recall, time_us_per_query, distance_comp_per_query, hops_per_query);
        if (recall > 1.0)
        {
            fmt::print("recall {} \t time_us_per_query {}us\n", recall, time_us_per_query);
            break;
        }
    }
}

int main()
{
    fmt::print("Testing ...\n");

    #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
    #endif

    size_t query_size = 10000;
    size_t base_size = 1000000;
    size_t top_k = 100;
    size_t vecdim = 128;
    size_t threads = 1;

    int efConstruction = 500;
    int M = 24;


    fmt::print("Testing  ...\n");
    
    auto data_path = std::filesystem::path(DATA_PATH);
    fmt::print("Data dir  {} \n", data_path.string().c_str());

    const auto path_query = (data_path / "SIFT1M/sift_query.fvecs").string();
    const auto path_groundtruth = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    const auto path_basedata = (data_path / "SIFT1M/sift_base.fvecs").string();
    char path_index[1024];
    const auto path_index_template = (data_path / "hnsw/sift1m_ef_%d_M_%d.hnsw").string();
    sprintf(path_index, path_index_template.c_str(), efConstruction, M);

    auto ground_truth = ivecs_read(path_groundtruth.c_str(), top_k, query_size);
    auto query_features = fvecs_read(path_query.c_str(), vecdim, query_size);
    auto base_features = fvecs_read(path_basedata.c_str(), vecdim, base_size);

    hnswlib::L2Space l2space(vecdim);
    hnswlib::HierarchicalNSW<float>* appr_alg;
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
        appr_alg->addPoint((void*)(base_features), (size_t)j1);

        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 10000;

        omp_set_num_threads(threads);
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
                              << " Mem: " << getCurrentRSS() / 1000000 << " Mb" << std::endl;
                    stopw.reset();
                }
            }
            appr_alg->addPoint((void*)(base_features + vecdim * j1), (size_t)j2);
        }

        std::cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds" << std::endl;
        appr_alg->saveIndex(path_index);
    }

    // test ground truth
    size_t k = 100;  // k at test time
    fmt::print("Parsing gt:\n");
    auto answer = get_ground_truth(ground_truth, query_size, k);
    fmt::print("Loaded gt:\n");
    for (int i = 0; i < 1; i++) test_vs_recall(*appr_alg, query_features, answer, vecdim, k);
    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);

    fmt::print("Test ok\n");

    return 0;
}