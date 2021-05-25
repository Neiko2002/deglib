#include <fmt/core.h>
#include <random>

#include "benchmark.h"
#include "graph.h"

int main() {
    fmt::print("Testing ...\n");

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif

    auto data_path = std::filesystem::path(DATA_PATH);

    // load an existing graph
    // auto graph_file = std::filesystem::path("k24nns_128D_L2_Path10_Rnd3+3Improve_AddK20Eps0.2_ImproveK20Eps0.025_WorstEdge0_cpp.graph");
    // fmt::print("Data dir  {} \n", data_path.string().c_str());
    // const auto graph = deglib::benchmark::load_sift1m_sizebounded_graph(data_path, graph_file);
    // const auto graph = deglib::benchmark::load_sift1m_readonly_graph(data_path, graph_file);

    // create a new graph
    const uint8_t edges_per_node = 24;
    auto repository = deglib::load_static_repository((data_path / "SIFT1M/sift_base.fvecs").string().c_str());
    const auto dims = repository.dims();
    const uint32_t max_node_count = repository.size();
    const auto feature_space = deglib::L2Space(dims);
    auto graph = deglib::graph::SizeBoundedGraph(max_node_count, edges_per_node, feature_space);

    auto rnd = std::mt19937(7); 
    const uint32_t extend_k = 20;
    const uint32_t improve_k = 20;
    const uint32_t max_path_length = 10;
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, extend_k, improve_k, max_path_length, rnd);

    for (uint32_t label = 0; label < repository.size(); label++) {
        builder.addEntry(label, reinterpret_cast<const std::byte*>(repository.getFeature(label)));
    }

    // start the build process
    auto val = builder.build();

    fmt::print("val {} \n", val);

    // const auto path_query_repository = (data_path / "SIFT1M/sift_query.fvecs").string();
    // const auto query_repository = deglib::load_static_repository(path_query_repository.c_str());
    // fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());
    // const auto path_query_groundtruth = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    // size_t dims;
    // size_t count;
    // const auto ground_truth_f = deglib::fvecs_read(path_query_groundtruth.c_str(), dims, count);
    // const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    // fmt::print("{} ground truth {} dimensions \n", count, dims);
    // deglib::benchmark::test_graph(graph, query_repository, ground_truth);

    fmt::print("Test OK\n");
    return 0;
}