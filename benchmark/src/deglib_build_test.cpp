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
    const uint8_t extend_k = 20; 
    const float extend_eps = 0.20;
    const uint8_t improve_k = 20;
    const float improve_eps = 0.025;
    const uint8_t max_path_length = 10;
    const uint32_t swap_tries = 3;
    const uint32_t additional_swap_tries = 3;
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, extend_k, extend_eps, improve_k, improve_eps, max_path_length, swap_tries, additional_swap_tries, rnd);

    // provide all features to the graph builder at once. In an online system this will be called 
    for (uint32_t label = 0; label < repository.size(); label++) {
        auto feature = reinterpret_cast<const std::byte*>(repository.getFeature(label));
        auto feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};
        builder.addEntry(label, std::move(feature_vector));
    }
    
    // check the integrity of the graph during the graph build process
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {

        if(status.step % 10000 == 0) {
            auto quality = deglib::analysis::calc_graph_quality(graph);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            fmt::print("step {}, added {}, deleted {}, improved {}, tries {}, connected {}, quality {} \n", status.step, status.added, status.deleted, status.improved, status.tries, connected, quality);
        }

        // check the graph from time to time
        if(status.step % 10000 == 0) {
            auto valid = deglib::analysis::check_graph_validation(graph, uint32_t(status.added - status.deleted), true);
            if(valid == false) {
                builder.stop();
                fmt::print("Invalid graph, build process is stopped");
            } 
        }
    };

    // start the build process
    builder.build(improvement_callback, false);


    


    const auto path_query_repository = (data_path / "SIFT1M/sift_query.fvecs").string();
    const auto query_repository = deglib::load_static_repository(path_query_repository.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());
    const auto path_query_groundtruth = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    size_t dims_out;
    size_t count_out;
    const auto ground_truth_f = deglib::fvecs_read(path_query_groundtruth.c_str(), dims_out, count_out);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", count_out, dims_out);
    deglib::benchmark::test_graph(graph, query_repository, ground_truth);

    fmt::print("Test OK\n");
    return 0;
}