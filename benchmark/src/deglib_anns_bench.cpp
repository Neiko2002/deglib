#include <random>

#include <fmt/core.h>
#include <cmath>

#include "benchmark.h"
#include "deglib.h"

int main() {
    fmt::print("Testing ...\n");

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif


    const auto data_path = std::filesystem::path(DATA_PATH);

    // ------------------------------------ SIFT1M -------------------------
    const uint32_t k = 100; 
    const uint32_t repeat_test = 1;

    const auto path_query_repository = (data_path / "SIFT1M" / "sift_query.fvecs").string();
    const auto path_query_groundtruth = (data_path / "SIFT1M" / "sift_groundtruth.ivecs").string();
    const auto after_improvements = std::vector<uint32_t>{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 20000000};


    // ------------------------------------ audio -------------------------
    // const uint32_t k = 20; 
    // const uint32_t repeat_test = 50;

    // const auto path_query_repository = (data_path / "audio" / "audio_query.fvecs").string();
    // const auto path_query_groundtruth = (data_path / "audio" / "audio_groundtruth.ivecs").string();
    // const auto after_improvements = std::vector<uint32_t>{10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 500000, 1000000, 1500000, 2000000, 3000000, 4000000, 6000000, 10000000};

    for (uint32_t i = 0; i < after_improvements.size(); i++) {
        const auto base_size = after_improvements[i]; //100000 * i;

        // SIFT1M
        const auto graph_file = (data_path / "deg" / "average_neighbor_rank2" / ("128D_L2_K30_RndAdd_SwapK30-0StepEps0.001LowPath5Rnd0+0_it"+std::to_string(base_size)+".deg")).string();

        // audio
        // const auto graph_file = (data_path / "deg" / "average_neighbor_rank" / ("192D_L2_K20_RndAdd_SwapK20-0StepEps0.001LowPath5Rnd0+0_it"+std::to_string(base_size)+".deg")).string();
     

        fmt::print("Load graph {} \n", graph_file);
        fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
        fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
        const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
        fmt::print("Graph with {} vertices \n", graph.size());
        fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
        fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);


        // load the test data and run several ANNS on the graph   
        const auto query_repository = deglib::load_static_repository(path_query_repository.c_str());
        fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());

        size_t ground_truth_dims;
        size_t ground_truth_count;
        const auto ground_truth_f = deglib::fvecs_read(path_query_groundtruth.c_str(), ground_truth_dims, ground_truth_count);
        const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
        fmt::print("{} ground truth {} dimensions \n", ground_truth_count, ground_truth_dims);

        // for (uint32_t i = 5; i < 11; i++) {
            auto k_test = k; //pow(2, i);
            fmt::print("Test with k={} and base size={}\n", k_test, base_size);
            deglib::benchmark::test_graph_anns(graph, query_repository, ground_truth, (uint32_t) ground_truth_dims, repeat_test, k_test);
        // }
    }

    fmt::print("Test OK\n");
    return 0;
}