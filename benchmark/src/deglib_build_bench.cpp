#include <random>
#include <chrono>

#include <fmt/core.h>

#include "benchmark.h"
#include "deglib.h"




void rng_optimize_graph(const std::string& graph_file, const std::string& optimized_graph_file) {

    auto rnd = std::mt19937(7); 

    fmt::print("Load graph {} \n", graph_file);
    auto graph = deglib::graph::load_sizebounded_graph(graph_file.c_str());
    fmt::print("Graph with {} nodes \n", graph.size());
    
    // auto optimizer = deglib::builder::EvenRegularGraphOptimizer(graph, rnd);
    // optimizer.removeNonRngEdges();
    // // optimizer.reduceNonRngEdges();

    // graph.saveGraph(optimized_graph_file.c_str());

    // try to optimize all not RNG conform edges 
    const uint8_t improve_k = 30;
    const float improve_eps = 0.001f;
    const bool improve_highLID = false;
    const uint8_t improve_step_factor = 0;
    const uint8_t max_path_length = 5; 
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, 0, 0, true, improve_k, improve_eps, improve_highLID, improve_step_factor, max_path_length, 0, 0);
    builder.optimizeRNGUnconformEdges();

    // store the graph
    graph.saveGraph(optimized_graph_file.c_str());
    // graph.saveGraph((optimized_graph_file+"opt_again.deg").c_str());
}

void naive_reorder_graph(const std::string& graph_file_in, const std::string& order_file, const std::string& graph_file_out) {
   
    // load an existing graph
    fmt::print("Load graph {} \n", graph_file_in);
    auto graph = deglib::graph::load_sizebounded_graph(graph_file_in.c_str());
    const auto max_node_count = graph.size();
    const auto edges_per_node = graph.getEdgesPerNode();

    fmt::print("Find a simple new order of the graph nodes \n");
    auto new_order = std::vector<uint32_t>(max_node_count);
    new_order.clear();
    auto checked_ids = std::vector<bool>(max_node_count);
    for (uint32_t i = 0; i < max_node_count; i++) {
        if(checked_ids[i] == false) {
            new_order.emplace_back(i);
            auto neighbors = graph.getNeighborIndices(i);
            for (uint32_t e = 0; e < edges_per_node; e++) {
                auto neighbor = neighbors[e];
                if(checked_ids[neighbor] == false) 
                    new_order.emplace_back(neighbor);
                checked_ids[neighbor] = true;
            }
            checked_ids[i] = true;
        }
    }

    // reorder the nodes in the graph
    fmt::print("Reorder the nodes in the graph \n");
    graph.reorderNodes(new_order);

    // store the graph
    fmt::print("Store new graph {} \n", graph_file_out);
    graph.saveGraph(graph_file_out.c_str());

    fmt::print("Store order file {} \n", order_file);
    auto out = std::ofstream(order_file, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(new_order.data()), new_order.size() * sizeof(uint32_t));    
    out.close();
}

void reorder_graph(const std::string graph_file_in, const std::string order_file, const std::string graph_file_out) {

    // load an existing graph
    fmt::print("Load graph {} \n", graph_file_in);
    auto graph = deglib::graph::load_sizebounded_graph(graph_file_in.c_str());
    const auto max_node_count = graph.size();

    // the order in which the node based on their labels should be ordered
    fmt::print("Load reorder file {} \n", order_file);
    std::error_code ec{};
    auto ifstream = std::ifstream(order_file.c_str(), std::ios::binary);
    auto order_array = std::make_unique<uint32_t[]>(max_node_count);
    ifstream.read(reinterpret_cast<char*>(order_array.get()), max_node_count * sizeof(uint32_t));
    ifstream.close();

    // reorder the nodes in the graph
    fmt::print("Reorder the nodes in the graph \n");
    auto order_ptr = order_array.get();
    auto new_order = std::vector<uint32_t>(order_ptr, order_ptr + max_node_count);
    graph.reorderNodes(new_order);

    // store the graph
    fmt::print("Store new graph {} \n", graph_file_out);
    graph.saveGraph(graph_file_out.c_str());
}

/**
 * Load the SIFT repository and create a dynamic exploratino graph with it.
 * Store the graph in the graph file.
 */
void create_graph(const std::string repository_file, const std::string order_file, const std::string graph_file) {
    
    auto rnd = std::mt19937(7); 
    const uint8_t extend_k = 60; // should be greater equals edges_per_node
    const float extend_eps = 0.2f;
    const bool extend_highLID = true;
    const uint8_t improve_k = 30;
    const float improve_eps = 0.001f;
    const bool improve_highLID = false;
    const uint8_t improve_step_factor = 0;
    const uint8_t max_path_length = 5; 
    const uint32_t swap_tries = 1;
    const uint32_t additional_swap_tries = 1;

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
    
    // the order in which the features should be used
    // std::error_code ec{};
    // auto ifstream = std::ifstream(order_file.c_str(), std::ios::binary);
    // auto order_array = std::make_unique<uint32_t[]>(max_node_count);
    // ifstream.read(reinterpret_cast<char*>(order_array.get()), max_node_count * sizeof(uint32_t));
    // ifstream.close();

    // provide all features to the graph builder at once. In an online system this will be called 
    for (uint32_t i = 0; i < repository.size(); i++) {
        auto label = i;
        // auto label = order_array[i];
        auto feature = reinterpret_cast<const std::byte*>(repository.getFeature(label));
        auto feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};
        builder.addEntry(label, std::move(feature_vector));
    }
    fmt::print("start building \n");
    
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
    // const auto path_query_repository = (data_path / "query.fvecs").string();
    // const auto path_query_groundtruth = (data_path / "query_gt.ivecs").string();
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

    // //SIFT1M
    const auto repository_file = (data_path / "SIFT1M/sift_base.fvecs").string();
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.rng_optimized.deg").string();
    const auto graph_file =             (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd1+1_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_swap_rng_opt.swap_pref_none_rng.deg").string();
    const auto optimized_graph_file =   (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd3+3_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.swap_pref_none_rng.rng_opt.deg").string();

    // ----------------------------------------------------------------------------------------------
    // --------------- TODO: Anstatt 5mio non-rng conform edges zu testen, werden die 5 schlechtesten edges eines jeden Knoten getestet
    // --------------- TODO: Der hohe Factor für non-RNG edges kann auch beim Swappen eingebaut werden
    // ----------------------------------------------------------------------------------------------

    const auto order_file = (data_path / "SIFT1M/sift_base_initial_order.int").string();
    // // const auto order_file = (data_path / "SIFT1M/sift_base_order_naive.int").string();
    // // const auto order_file = (data_path / "SIFT1M/sift_base_order359635264.int").string();
    // //  const auto order_file = (data_path / "SIFT1M/sift_base_order232076720.int").string();
    // // const auto order_file = (data_path / "SIFT1M/sift_base_order223619312.int").string();


    // GLOVE
    //const auto repository_file = (data_path / "glove-100/glove-100_base.fvecs").string();
    //const auto graph_file = (data_path / "deg" / "100D_L2_K30_AddK30Eps0.2HighRNGSelectOptional_ImproveK30-0StepEps0.001Low_Path5_Rnd0+0_realHighLows_improveNonPerfectEdges_noLoopDetection.deg").string();
    // const auto graph_file = (data_path / "deg" / "100D_L2_K22_AddK30Eps0.2High_ImproveK30-0StepEps0.001Low_Path5_Rnd8+8_realHighLows_improveNonPerfectEdges_noLoopDetection.deg").string();


    // 2DGraph
    // const auto repository_file = (data_path / "base.fvecs").string();
    // const auto graph_file = (data_path / "L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_swap_rng_opt.deg").string();
    // const auto optimized_graph_file = (data_path / "L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_swap_rng_opt.remove_non_rng_edges.deg").string();

    // load the SIFT base features and creates a DEG graph with them. The graph is than stored on the drive.
    // create_graph(repository_file, order_file, graph_file);

    // loads the graph from the drive and test it against the SIFT query data
    test_graph(data_path, graph_file, repeat_test, test_k);


    // // load the SIFT base features and creates a DEG graph with them. The graph is than stored on the drive.
    // rng_optimize_graph(graph_file, optimized_graph_file);

    // // loads the graph from the drive and test it against the SIFT query data
    // test_graph(data_path, optimized_graph_file, repeat_test, test_k);






    // const auto graph_file_out = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0_postOrdered232076720.test.deg").string();
    // reorder_graph(graph_file, order_file, graph_file_out);
    // naive_reorder_graph(graph_file, order_file,  graph_file_out);
    // test_graph(data_path, graph_file_out, repeat_test, test_k);

    fmt::print("Test OK\n");
    return 0;
}