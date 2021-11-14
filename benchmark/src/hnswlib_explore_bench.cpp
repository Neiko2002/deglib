#include <omp.h>
#include <fmt/core.h>
#include <tsl/robin_set.h>

#include <algorithm>

#include "hnswlib.h"
#include "hnsw/utils.h"
#include "stopwatch.h"

static std::vector<tsl::robin_set<size_t>> get_ground_truth(const uint32_t* ground_truth, const size_t ground_truth_size, const size_t ground_truth_dims, const size_t k)
{
    auto answers = std::vector<tsl::robin_set<size_t>>(ground_truth_size);
    for (int i = 0; i < ground_truth_size; i++)
    {
        auto& gt = answers[i];
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) 
            gt.insert(ground_truth[ground_truth_dims * i + j]);
    }

    return answers;
}

static float test_approx_explore(const hnswlib::HierarchicalNSW<float>& appr_alg, const float* query_repository,
                         const std::vector<tsl::robin_set<size_t>>& ground_truth, const size_t query_dims,
                         const std::vector<std::vector<uint32_t>>& entry_node_indizies,
                         const uint32_t k, const uint32_t max_distance_count)
{
    size_t correct = 0;
    size_t total = 0;
    for (int i = 0; i < ground_truth.size(); i++) {
        auto entry_node = entry_node_indizies[i][0];

        auto gt = ground_truth[i];
        auto result_queue = appr_alg.explore(entry_node, k, max_distance_count);

        total += gt.size();
        while (result_queue.empty() == false)
        {
            if (gt.find(result_queue.top().second) != gt.end()) correct++;
            result_queue.pop();
        }
    }

    return 1.0f * correct / total;
}


static void test_vs_recall_explore(hnswlib::HierarchicalNSW<float>& appr_alg, const float* query_repository,
                           const std::vector<tsl::robin_set<size_t>>& ground_truth, const size_t query_dims,
                           const uint32_t* entry_nodes, const uint32_t entry_node_dims,
                           const size_t k)
{
    // reproduceable entry point for the graph search
    auto entry_node_indizies = std::vector<std::vector<uint32_t>>();
    for (size_t i = 0; i < ground_truth.size(); i++) {
        auto entry_node = std::vector<uint32_t>(entry_nodes + i * entry_node_dims, entry_nodes + (i+1) * entry_node_dims);
        entry_node_indizies.emplace_back(entry_node);
    }

    // try different k values
    uint32_t steps = 100;
    for (uint32_t i = 0; i <= steps; i++) {
        const auto max_distance_count = k + (k/10 * i);

        appr_alg.setEf(k*2);
        appr_alg.metric_distance_computations = 0;
        appr_alg.metric_hops = 0;

        auto stopw = StopW();
        auto recall = test_approx_explore(appr_alg, query_repository, ground_truth, query_dims, entry_node_indizies, k, max_distance_count);
        auto time_us_per_query = stopw.getElapsedTimeMicro() / ground_truth.size();

        float distance_comp_per_query = appr_alg.metric_distance_computations / (1.0f * ground_truth.size());
        float hops_per_query = appr_alg.metric_hops / (1.0f * ground_truth.size());

        fmt::print("max_distance_count {} \t recall {} \t time_us_per_query {}us, avg distance computations {}, avg hops {}\n", max_distance_count, recall, time_us_per_query, distance_comp_per_query, hops_per_query);
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
    int threads = 1;
    
    int efConstruction = 500;
    int M = 24;
    size_t k = 1000;  // k at test time

    fmt::print("Testing  ...\n");
    
    auto data_path = std::filesystem::path(DATA_PATH);
    fmt::print("Data dir  {} \n", data_path.string().c_str());

    const auto path_query = (data_path / "SIFT1M/sift_explore_query.fvecs").string();
    const auto path_groundtruth = (data_path / "SIFT1M/sift_explore_ground_truth.ivecs").string();
    const auto path_entry = (data_path / "SIFT1M/sift_explore_entry_node.ivecs").string();
    const auto path_basedata = (data_path / "SIFT1M/sift_base.fvecs").string();

    char path_index[1024];
    const auto path_index_template = (data_path / "hnsw/sift1m_ef_%d_M_%d.hnsw").string();
    std::sprintf(path_index, path_index_template.c_str(), efConstruction, M);

    auto ground_truth = ivecs_read(path_groundtruth.c_str(), top_k, query_size);
    auto query_features = fvecs_read(path_query.c_str(), vecdim, query_size);
    auto base_features = fvecs_read(path_basedata.c_str(), vecdim, base_size);

    hnswlib::L2Space l2space(vecdim);
    hnswlib::HierarchicalNSW<float>* appr_alg;
    if (exists_test(path_index) == false) {
        std::cerr << "Loading index from " << path_index << ":" << std::endl;
        abort();
    }

    std::cout << "Loading index from " << path_index << ":" << std::endl;
    appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, path_index, false);
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb" << std::endl;

    // test ground truth
    fmt::print("Parsing gt:\n");
    auto answer = get_ground_truth(ground_truth, query_size, top_k, k);
    fmt::print("Loaded gt:\n");

    size_t entry_node_dims;
    size_t entry_node_count;
    const auto entry_node = ivecs_read(path_entry.c_str(), entry_node_dims, entry_node_count);
    fmt::print("{} exploration entry node {} dimensions \n", entry_node_count, entry_node_dims);
    fmt::print("Explore for {} neighbors \n", k);

    test_vs_recall_explore(*appr_alg, query_features, answer, vecdim, entry_node, (uint32_t) entry_node_dims, k);

    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Test ok\n");

    return 0;
}