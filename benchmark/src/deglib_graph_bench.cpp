#include <random>

#include <fmt/core.h>

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

    const auto repeat_test = 3;
    const auto data_path = std::filesystem::path(DATA_PATH);

    // load an existing graph
    const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2_ImproveK20Eps0.02.deg").string();
    fmt::print("Load graph {} \n", graph_file);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());

    // test the graph
    const auto path_query_repository = (data_path / "SIFT1M/sift_query.fvecs").string();
    const auto query_repository = deglib::load_static_repository(path_query_repository.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());
    const auto path_query_groundtruth = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    size_t dims;
    size_t count;
    const auto ground_truth_f = deglib::fvecs_read(path_query_groundtruth.c_str(), dims, count);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", count, dims);
    deglib::benchmark::test_graph(graph, query_repository, ground_truth, (uint32_t) dims, repeat_test);

    fmt::print("Test OK\n");
    return 0;
}