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
    
    auto rnd = std::mt19937(7); 
    const uint8_t extend_k = 60; // should be greater equals edges_per_node
    const float extend_eps = 0.2f;
    const bool extend_highLID = true;
    const uint8_t improve_k = 30;
    const float improve_eps = 0.001f;
    const bool improve_highLID = false;
    const uint8_t improve_step_factor = 0;
    const uint8_t max_path_length = 5; 
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;

    // create a new graph
    const uint8_t edges_per_node = 30;
    const deglib::Metric metric = deglib::Metric::L2;
    auto repository = deglib::load_static_repository(repository_file.c_str());
    const auto dims = repository.dims();
    const uint32_t max_node_count = uint32_t(repository.size());
    const auto feature_space = deglib::FloatSpace(dims, metric);
    auto graph = deglib::graph::SizeBoundedGraph(max_node_count, edges_per_node, feature_space);

    // create a graph builder to add nodes to the new graph and improve its edges
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, extend_k, extend_eps, extend_highLID, improve_k, improve_eps, improve_highLID, improve_step_factor, max_path_length, swap_tries, additional_swap_tries);
    // auto builder = deglib::builder::EvenRegularGraphBuilderExperimental(graph, rnd, extend_k, extend_eps, extend_highLID, improve_k, improve_eps, improve_highLID, improve_step_factor, max_path_length, swap_tries, additional_swap_tries);
    
    // provide all features to the graph builder at once. In an online system this will be called 
    for (uint32_t label = 0; label < repository.size(); label++) {
        auto feature = reinterpret_cast<const std::byte*>(repository.getFeature(label));
        auto feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};
        builder.addEntry(label, std::move(feature_vector));
    }
    
    // check the integrity of the graph during the graph build process
    const auto log_after = 10000;
    auto start = std::chrono::system_clock::now();
    uint64_t duration_ms = 0;
    bool valid = true;
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {

        if(status.added % log_after == 0) {
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph);
            auto weight_histogram_sorted = deglib::analysis::calc_edge_weight_histogram(graph, true);
            auto weight_histogram = deglib::analysis::calc_edge_weight_histogram(graph, false);
            auto valid_weights = deglib::analysis::check_graph_weights(graph);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms / 1000;
            fmt::print("{:7} nodes, {:5}s, {:8} / {:8} improv, Q: {:4.2f} -> Sorted:{:.1f}, InOrder:{:.1f}, {} connected & {}\n", 
                        status.added, duration, status.improved, status.tries, avg_edge_weight, fmt::join(weight_histogram_sorted, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid");
            start = std::chrono::system_clock::now();
        }

        // check the graph from time to time
        if(status.added % log_after == 0) {
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count());
            valid = deglib::analysis::check_graph_validation(graph, uint32_t(status.added - status.deleted), true);
            if(valid == false) {
                builder.stop();
                fmt::print("Invalid graph, build process is stopped\n");
            }
            start = std::chrono::system_clock::now();
        }
    };

    // start the build process
    builder.build(improvement_callback, false);

    // store the graph
    if(valid)
        graph.saveGraph(graph_file.c_str());
}

/**
 * Load the graph from the drive and test it against the SIFT query data.
 */
void test_graph(const std::filesystem::path data_path, const std::string graph_file, const uint32_t repeat, const uint32_t k) {
    const auto path_query_repository = (data_path / "SIFT1M/sift_query.fvecs").string();
    const auto path_query_groundtruth = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    //const auto path_query_repository = (data_path / "glove-100/glove-100_query.fvecs").string();
    //const auto path_query_groundtruth = (data_path / "glove-100/glove-100_groundtruth.ivecs").string();


    // load an existing graph
    fmt::print("Load graph {} \n", graph_file);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    // const auto graph = deglib::graph::load_sizebounded_graph(graph_file.c_str());
    
    // generall graph stats
    {
        const auto mutable_graph = deglib::graph::load_sizebounded_graph(graph_file.c_str());
        auto graph_memory = graph.getEdgesPerNode() * graph.size() * 8 / 1000000; // 4 bytes node id and 4 bytes for the weight
        auto avg_weight = deglib::analysis::calc_avg_edge_weight(mutable_graph);
        auto weight_histogram_ordered = deglib::analysis::calc_edge_weight_histogram(mutable_graph, true);
        auto weight_histogram = deglib::analysis::calc_edge_weight_histogram(mutable_graph, false);
        fmt::print("Graph memory {}mb during build, avg weight {:.2f} (every 10%, Sorted: {:.1f}, InOrder: {:.1f})\n", graph_memory, avg_weight, fmt::join(weight_histogram_ordered, " "), fmt::join(weight_histogram, " ")); 
    }

    const auto query_repository = deglib::load_static_repository(path_query_repository.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());

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

    const uint32_t repeat_test = 1;
    const uint32_t test_k = 100;
    const auto data_path = std::filesystem::path(DATA_PATH);

    //SIFT1M
    const auto repository_file = (data_path / "SIFT1M/sift_base.fvecs").string();
    const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.test1.deg").string();
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0NoImproveRNG.deg").string();


    // GLOVE
    //const auto repository_file = (data_path / "glove-100/glove-100_base.fvecs").string();
    //const auto graph_file = (data_path / "deg" / "100D_L2_K30_AddK30Eps0.2HighRNGSelectOptional_ImproveK30-0StepEps0.001Low_Path5_Rnd0+0_realHighLows_improveNonPerfectEdges_noLoopDetection.deg").string();
    // const auto graph_file = (data_path / "deg" / "100D_L2_K22_AddK30Eps0.2High_ImproveK30-0StepEps0.001Low_Path5_Rnd8+8_realHighLows_improveNonPerfectEdges_noLoopDetection.deg").string();

    // load the SIFT base features and creates a DEG graph with them. The graph is than stored on the drive.
    create_graph(repository_file, graph_file);

    // loads the graph from the drive and test it against the SIFT query data
    test_graph(data_path, graph_file, repeat_test, test_k);

    fmt::print("Test OK\n");
    return 0;
}