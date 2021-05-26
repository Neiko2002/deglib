#include <fmt/core.h>
#include <random>

#include "benchmark.h"
#include "graph.h"
#include "analysis.h"

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

    // create a new graph
    const uint8_t edges_per_node = 24;
    auto repository = deglib::load_static_repository((data_path / "SIFT1M/sift_base.fvecs").string().c_str());
    const auto dims = repository.dims();
    const uint32_t max_node_count = uint32_t(repository.size());
    const auto feature_space = deglib::L2Space(dims);
    auto graph = deglib::graph::SizeBoundedGraph(max_node_count, edges_per_node, feature_space);

    // create a graph builder to add nodes to the new graph and improve its edges
    auto rnd = std::mt19937(7); 
    const uint32_t extend_k = 20;
    const uint32_t improve_k = 20;
    const uint32_t max_path_length = 10;
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, extend_k, improve_k, max_path_length, rnd);

    // provide all features to the graph builder at once. In an online system this will be called 
    for (uint32_t label = 0; label < repository.size(); label++) {
        auto feature = reinterpret_cast<const std::byte*>(repository.getFeature(label));
        auto feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};
        builder.addEntry(label, std::move(feature_vector));
    }
    
    // check the integrity of the graph during the graph build process
    const auto improvement_callback = [&](uint64_t step, uint64_t added, uint64_t deleted, uint64_t improved) {
        fmt::print("step {}, added {}, deleted {}, improved {} \n", step, added, deleted, improved);

        // check the graph from time to time
        if(step % 100 == 0) {
            auto valid = deglib::analysis::validation_check(graph, uint32_t(added - deleted));
            if(valid == false) {
                builder.stop();
                fmt::print("Invalid graph, build process is stopped");
            }
        }
    };

    // start the build process
    builder.build(improvement_callback, false);


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