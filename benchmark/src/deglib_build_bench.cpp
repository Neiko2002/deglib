#include <random>
#include <chrono>

#include <fmt/core.h>

#include "benchmark.h"
#include "deglib.h"


/**
 * Load the SIFT repository and create a dynamic exploratino graph with it.
 * Store the graph in the graph file.
 */
void create_graph(const std::string repository_file, const std::string graph_file) {

    // create a new graph
    const uint8_t edges_per_node = 40;
    auto repository = deglib::load_static_repository(repository_file.c_str());
    const auto dims = repository.dims();
    const uint32_t max_node_count = uint32_t(repository.size());
    const auto feature_space = deglib::L2Space(dims);
    auto graph = deglib::graph::SizeBoundedGraph(max_node_count, edges_per_node, feature_space);

    // create a graph builder to add nodes to the new graph and improve its edges
    // best G24: k24nns_128D_L2_AddK24Eps0.2_ImproveK24Eps0.02_ImproveExtK24-2StepEps0.02_Path10_Rnd15+15.deg
    // best G40: k40nns_128D_L2_AddK40Eps0.1_ImproveK40Eps0.02_ImproveExtK40-2StepEps0.02_Path12_Rnd3+3.deg
    auto rnd = std::mt19937(7); 
    const uint8_t extend_k = 40; // should always be >= edges_per_node
    const float extend_eps = 0.1f;
    const uint8_t improve_k = 40;
    const float improve_eps = 0.02f;
    const uint8_t improve_extended_k = 40;
    const float improve_extended_eps = 0.02f;
    const uint8_t improve_extended_step_factor = 2;
    const uint8_t max_path_length = 12; 
    const uint32_t swap_tries = 7;
    const uint32_t additional_swap_tries = 7;
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, extend_k, extend_eps, improve_k, improve_eps, improve_extended_k, improve_extended_eps, improve_extended_step_factor, max_path_length, swap_tries, additional_swap_tries);

    // provide all features to the graph builder at once. In an online system this will be called 
    for (uint32_t label = 0; label < repository.size(); label++) {
        auto feature = reinterpret_cast<const std::byte*>(repository.getFeature(label));
        auto feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};
        builder.addEntry(label, std::move(feature_vector));
    }
    
    // check the integrity of the graph during the graph build process
    const auto log_after = 10000;
    const auto start = std::chrono::system_clock::now();
    auto last_status = deglib::builder::BuilderStatus{};
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {

        if(status.added % log_after == 0) {
            auto quality = deglib::analysis::calc_graph_quality(graph);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count());
            auto avg_improv = uint32_t((status.improved - last_status.improved) / log_after);
            auto avg_tries = uint32_t((status.tries - last_status.tries) / log_after);
            fmt::print("{:7} elements, in {:5}s, with {:8} / {:8} improvements (avg {:2}/{:3}), quality {:4.2f}, connected {} \n", 
                        status.added, duration, status.improved, status.tries, avg_improv, avg_tries, quality, connected);
        }

        // check the graph from time to time
        if(status.added % log_after == 0) {
            auto valid = deglib::analysis::check_graph_validation(graph, uint32_t(status.added - status.deleted), true);
            if(valid == false) {
                builder.stop();
                fmt::print("Invalid graph, build process is stopped");
            } 
        }

        last_status = status;
    };

    // start the build process
    builder.build(improvement_callback, false);

    // store the graph
    graph.saveGraph(graph_file.c_str());
}

/**
 * Load the graph from the drive and test it against the SIFT query data.
 */
void test_graph(const std::filesystem::path data_path, const std::string graph_file, const uint32_t repeat, const uint32_t k) {

    // load an existing graph
    fmt::print("Load graph {} \n", graph_file);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());

    const auto path_query_repository = (data_path / "SIFT1M/sift_query.fvecs").string();
    const auto query_repository = deglib::load_static_repository(path_query_repository.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());
    const auto path_query_groundtruth = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    size_t dims_out;
    size_t count_out;
    const auto ground_truth_f = deglib::fvecs_read(path_query_groundtruth.c_str(), dims_out, count_out);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", count_out, dims_out);
    deglib::benchmark::test_graph_anns(graph, query_repository, ground_truth, (uint32_t)dims_out, repeat, k);
}

int main() {
    fmt::print("Testing ...\n");

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif

    const uint32_t repeat_test = 3;
    const uint32_t test_k = 100;
    const auto data_path = std::filesystem::path(DATA_PATH);
    const auto repository_file = (data_path / "SIFT1M/sift_base.fvecs").string();
    const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k40nns_128D_L2_AddK40Eps0.1_ImproveK40Eps0.02_ImproveExtK40-2StepEps0.02_Path12_Rnd5+5.deg").string();

    // load the SIFT base features and creates a DEG graph with them. The graph is than stored on the drive.
    create_graph(repository_file, graph_file);

    // loads the graph from the drive and test it against the SIFT query data
    test_graph(data_path, graph_file, repeat_test, test_k);

    fmt::print("Test OK\n");
    return 0;
}