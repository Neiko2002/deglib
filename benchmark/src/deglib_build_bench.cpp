#include <random>
#include <chrono>

#include <fmt/core.h>

#include "benchmark.h"
#include "deglib.h"




void rng_optimize_graph(const std::string& graph_file, const std::string& optimized_graph_file) {

    auto rnd = std::mt19937(7); 

    fmt::print("Load graph {} \n", graph_file);
    auto graph = deglib::graph::load_sizebounded_graph(graph_file.c_str());
    fmt::print("Graph with {} vertices and {} non-RNG edges \n", graph.size(), deglib::analysis::calc_non_rng_edges(graph));
    
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

    fmt::print("The RNG optimized graph contains {} non-RNG edges\n", deglib::analysis::calc_non_rng_edges(graph)); 
}

void reorder_graph(const std::string graph_file_in, const std::string order_file, const std::string graph_file_out) {

    // load an existing graph
    fmt::print("Load graph {} \n", graph_file_in);
    auto graph = deglib::graph::load_sizebounded_graph(graph_file_in.c_str());
    const auto max_vertex_count = graph.size();

    // the order in which the vertex based on their labels should be ordered
    fmt::print("Load reorder file {} \n", order_file);
    std::error_code ec{};
    auto ifstream = std::ifstream(order_file.c_str(), std::ios::binary);
    auto order_array = std::make_unique<uint32_t[]>(max_vertex_count);
    ifstream.read(reinterpret_cast<char*>(order_array.get()), max_vertex_count * sizeof(uint32_t));
    ifstream.close();

    // reorder the vertices in the graph
    fmt::print("Reorder the vertices in the graph \n");
    auto order_ptr = order_array.get();
    auto new_order = std::vector<uint32_t>(order_ptr, order_ptr + max_vertex_count);
    graph.reorderNodes(new_order);

    // store the graph
    fmt::print("Store new graph {} \n", graph_file_out);
    graph.saveGraph(graph_file_out.c_str());
}

/**
 * Load the data repository and create a dynamic exploratino graph with it.
 * Store the graph in the graph file.
 */
void create_graph(const std::string repository_file, const std::string order_file, const std::string graph_file) {
    
    auto rnd = std::mt19937(7);  // default 7
    const int weight_scale = 1; // SIFT+Glove+enron+crawl=1 UQ-V=100000
    const uint8_t edges_per_vertex = 30;
    const deglib::Metric metric = deglib::Metric::L2;
    const uint8_t extend_k = 30; // should be greater or equals to edges_per_vertex
    const float extend_eps = 0.2f;
    const bool extend_highLID = true;
    const uint8_t improve_k = 30;
    const float improve_eps = 0.001f;
    const bool improve_highLID = false;
    const uint8_t improve_step_factor = 0;
    const uint8_t max_path_length = 5; 
    const uint32_t swap_tries = 0;
    const uint32_t additional_swap_tries = 0;

    // load data
    fmt::print("Load Data \n");
    auto repository = deglib::load_static_repository(repository_file.c_str());   
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading data\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // create a new graph
    fmt::print("Setup empty graph with {} vertices in {}D feature space\n", repository.size(), repository.dims());
    const auto dims = repository.dims();
    const uint32_t max_vertex_count = uint32_t(repository.size());
    const auto feature_space = deglib::FloatSpace(dims, metric);
    auto graph = deglib::graph::SizeBoundedGraph(max_vertex_count, edges_per_vertex, feature_space);
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup empty graph\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // create a graph builder to add vertices to the new graph and improve its edges
    fmt::print("Start graph builder \n");   
    //auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, extend_k, extend_eps, extend_highLID, improve_k, improve_eps, improve_highLID, improve_step_factor, max_path_length, swap_tries, additional_swap_tries);
    auto builder = deglib::builder::EvenRegularGraphBuilderExperimental(graph, rnd, extend_k, extend_eps, extend_highLID, improve_k, improve_eps, improve_highLID, improve_step_factor, max_path_length, swap_tries, additional_swap_tries);
    
    // the order in which the features should be used
    // std::error_code ec{};
    // auto ifstream = std::ifstream(order_file.c_str(), std::ios::binary);
    // auto order_array = std::make_unique<uint32_t[]>(max_vertex_count);
    // ifstream.read(reinterpret_cast<char*>(order_array.get()), max_vertex_count * sizeof(uint32_t));
    // ifstream.close();

    // provide all features to the graph builder at once. In an online system this will be called 
    for (uint32_t i = 0; i < repository.size(); i++) {
        auto label = i;
        //auto label = order_array[i];
        auto feature = reinterpret_cast<const std::byte*>(repository.getFeature(label));
        auto feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};
        builder.addEntry(label, std::move(feature_vector));
    }
    repository.clear();
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup graph builder\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    fmt::print("Start building \n");    
    // check the integrity of the graph during the graph build process
    const auto log_after = 100000;
    auto start = std::chrono::steady_clock::now();
    uint64_t duration_ms = 0;
    bool valid = true;
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {

        if(status.added % log_after == 0 || status.added == max_vertex_count) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, weight_scale);
            auto weight_histogram_sorted = deglib::analysis::calc_edge_weight_histogram(graph, true, weight_scale);
            auto weight_histogram = deglib::analysis::calc_edge_weight_histogram(graph, false, weight_scale);
            auto valid_weights = deglib::analysis::check_graph_weights(graph);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = duration_ms / 1000;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} vertices, {:5}s, {:8} / {:8} improv, Q: {:4.2f} -> Sorted:{:.1f}, InOrder:{:.1f}, {} connected & {}, RSS {} & peakRSS {}\n", 
                        status.added, duration, status.improved, status.tries, avg_edge_weight, fmt::join(weight_histogram_sorted, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid", currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }
        else if(status.added % (log_after/10) == 0) {    
            duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
            auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, weight_scale);
            auto duration = duration_ms / 1000;
            auto currRSS = getCurrentRSS() / 1000000;
            auto peakRSS = getPeakRSS() / 1000000;
            fmt::print("{:7} vertices, {:5}s, {:8} / {:8} improv, Q: {:4.2f}, RSS {} & peakRSS {}\n", status.added, duration, status.improved, status.tries,avg_edge_weight, currRSS, peakRSS);
            start = std::chrono::steady_clock::now();
        }

        

        // check the graph from time to time
        // if(status.added % log_after == 0) {
        //     duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count());
        //     valid = deglib::analysis::check_graph_validation(graph, uint32_t(status.added - status.deleted), true);
        //     if(valid == false) {
        //         builder.stop();
        //         fmt::print("Invalid graph, build process is stopped\n");
        //     }
        //     start = std::chrono::system_clock::now();
        // }
    };

    // start the build process
    builder.build(improvement_callback, false);
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after building the graph in {} secs\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000, duration_ms / 1000);


   // store the graph
    if(valid)
        graph.saveGraph(graph_file.c_str());

    fmt::print("The graph contains {} non-RNG edges\n", deglib::analysis::calc_non_rng_edges(graph));
}

/**
 * Load the graph from the drive and test it against the SIFT query data.
 */
void test_graph(const std::filesystem::path data_path, const std::string graph_file, const uint32_t repeat, const uint32_t k) {

    // const auto path_query_repository = (data_path / "query.fvecs").string();
    // const auto path_query_groundtruth = (data_path / "query_gt.ivecs").string();
    const auto path_query_repository = (data_path / "SIFT1M/sift_query.fvecs").string();
    const auto path_query_groundtruth = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    // const auto path_query_repository = (data_path / "glove-100/glove-100_query.fvecs").string();
    // const auto path_query_groundtruth = (data_path / "glove-100/glove-100_groundtruth.ivecs").string();
    // const auto path_query_repository = (data_path / "uqv" / "uqv_query.fvecs").string();
    // const auto path_query_groundtruth = (data_path / "uqv" / "uqv_groundtruth.ivecs").string();
    // const auto path_query_repository = (data_path / "audio" / "audio_query.fvecs").string();
    // const auto path_query_groundtruth = (data_path / "audio" / "audio_groundtruth.ivecs").string();
    // const auto path_query_repository = (data_path / "enron" / "enron_query.fvecs").string();
    // const auto path_query_groundtruth = (data_path / "enron" / "enron_groundtruth.ivecs").string();
    // const auto path_query_repository = (data_path / "pixabay" / "pixabay_clipfv_query.fvecs").string();
    // const auto path_query_groundtruth = (data_path / "pixabay" / "pixabay_clipfv_groundtruth.ivecs").string();
    //   const auto path_query_repository = (data_path / "pixabay" / "pixabay_gpret_query.fvecs").string();
    // const auto path_query_groundtruth = (data_path / "pixabay" / "pixabay_gpret_groundtruth.ivecs").string();
    // const auto path_query_repository = (data_path / "crawl" / "crawl_query.fvecs").string();
    // const auto path_query_groundtruth = (data_path / "crawl" / "crawl_groundtruth.ivecs").string();

    // load an existing graph
    fmt::print("Load graph {} \n", graph_file);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading the graph\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // const auto graph = deglib::graph::load_readonly_irregular_graph(graph_file.c_str());
    // const auto graph = deglib::graph::load_sizebounded_graph(graph_file.c_str());
    
    // generall graph stats
    // {
    //     const auto mutable_graph = deglib::graph::load_sizebounded_graph(graph_file.c_str());
    //     auto graph_memory = graph.getEdgesPerNode() * graph.size() * 8 / 1000000; // 4 bytes vertex id and 4 bytes for the weight
    //     auto avg_weight = deglib::analysis::calc_avg_edge_weight(mutable_graph);
    //     auto weight_histogram_ordered = deglib::analysis::calc_edge_weight_histogram(mutable_graph, true);
    //     auto weight_histogram = deglib::analysis::calc_edge_weight_histogram(mutable_graph, false);
    //     auto non_rng_edge_count = deglib::analysis::calc_non_rng_edges(mutable_graph); 
    //     auto valid_weights = deglib::analysis::check_graph_weights(mutable_graph);
    //     auto connected = deglib::analysis::check_graph_connectivity(mutable_graph);
    //     fmt::print("Graph memory {}mb during build, avg weight {:.2f}, {:8} non-rng edges, (every 10%, Sorted: {:.1f}, InOrder: {:.1f}), {} connected & {}\n", graph_memory, avg_weight, non_rng_edge_count, fmt::join(weight_histogram_ordered, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid"); 
    // }

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

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb \n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);


    const auto data_path = std::filesystem::path(DATA_PATH);
    uint32_t repeat_test = 1;    
    uint32_t test_k = 100;


    // 2DGraph
    // test_k = 4;
    // const auto repository_file = (data_path / "base.fvecs").string();
    // const auto graph_file = (data_path / "L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd100+0_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.deg").string();
    // const auto optimized_graph_file = (data_path / "L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd100+0_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.remove_non_rng_edges.deg").string();

    //SIFT1M
    const auto repository_file =        (data_path / "SIFT1M/sift_base.fvecs").string();
    // const auto graph_file =             (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string();
    const auto graph_file =             (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0_postOrdered232076720.deg").string();
    const auto optimized_graph_file =   (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd3+2_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.non_rng_optimized.deg").string();

    // // const auto order_file = (data_path / "ignore").string();
    const auto order_file = (data_path / "SIFT1M/sift_base_initial_order.int").string();
    // // const auto order_file = (data_path / "SIFT1M/sift_base_order_naive.int").string();
    // // const auto order_file = (data_path / "SIFT1M/sift_base_order359635264.int").string();
    // // const auto order_file = (data_path / "SIFT1M/sift_base_order232076720.int").string();
    // // const auto order_file = (data_path / "SIFT1M/sift_base_order223619312.int").string();

    // GLOVE
    // const auto repository_file = (data_path / "glove-100/glove-100_base.fvecs").string();
    // const auto graph_file = (data_path / "deg" / "100D_L2_K30_AddK30Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string();
    // const auto optimized_graph_file = (data_path / "deg" / "100D_L2_K30_AddK30Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveNonRNGAndSecondHalfOfNonPerfectEdges.add_rngOptional_only28.rng_optimized_RNGSwapAtStep0.deg").string();

    // // UQ-V    
    // const auto repository_file      = (data_path / "uqv" / "uqv_base.fvecs").string();
    // const auto order_file           = (data_path / "uqv" / "sift_base_order232076720.int").string();
    // const auto graph_file           = (data_path / "deg" / "256D_L2_K20_AddK20Eps0.2Low_SwapK20-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string();
    // const auto optimized_graph_file = (data_path / "deg" / "256D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge_opt.deg").string();

    // // enron
    // test_k = 20;
    // repeat_test = 20;
    // const auto repository_file      = (data_path / "enron" / "enron_base.fvecs").string();
    // const auto order_file           = (data_path / "enron" / "sift_base_order232076720.int").string();
    // const auto graph_file           = (data_path / "deg" / "1369D_L2_K30_AddK60Eps0.3High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string();
    // const auto optimized_graph_file = (data_path / "deg" / "1369D_L2_K20_AddK20Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge_opt.deg").string();

    // pixabay clipfv
    // const auto repository_file      = (data_path / "pixabay" / "pixabay_clipfv_base.fvecs").string();
    // const auto order_file           = (data_path / "pixabay" / "sift_base_order232076720.int").string();
    // const auto graph_file           = (data_path / "deg" / "768D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string();
    // const auto optimized_graph_file = (data_path / "deg" / "768D_L2_K20_AddK20Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge_opt.deg").string();


    // // pixabay gpret
    // const auto repository_file      = (data_path / "pixabay" / "pixabay_gpret_base.fvecs").string();
    // const auto order_file           = (data_path / "pixabay" / "sift_base_order232076720.int").string();
    // const auto graph_file           = (data_path / "deg" / "1024D_L2_K30_AddK30Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string();
    // const auto optimized_graph_file = (data_path / "deg" / "1024D_L2_K20_AddK20Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge_opt.deg").string();

    // crawl
    // const auto repository_file      = (data_path / "crawl" / "crawl_base.fvecs").string();
    // const auto order_file           = (data_path / "crawl" / "sift_base_order232076720.int").string();
    // const auto graph_file           = (data_path / "deg" / "300D_L2_K40_AddK40Eps0.1Low.deg").string();
    // const auto optimized_graph_file = (data_path / "deg" / "300D_L2_K30_AddK30Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge_opt.deg").string();

    // audio
    // test_k = 20;
    // repeat_test = 10;
    // const auto repository_file      = (data_path / "audio" / "audio_base.fvecs").string();
    // const auto order_file           = (data_path / "audio" / "sift_base_order232076720.int").string();
    // const auto graph_file           = (data_path / "deg" / "192D_L2_K20_AddK20Eps0.2Low_SwapK20-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string();
    // const auto optimized_graph_file = (data_path / "deg" / "192D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge_opt.deg").string();


    // load the base features and creates a DEG graph with them. The graph is than stored on the drive.
    if(std::filesystem::exists(graph_file.c_str()) == false)
        create_graph(repository_file, order_file, graph_file);

    // loads the graph from the drive and test it against the SIFT query data
    test_graph(data_path, graph_file, repeat_test, test_k);


    // // load the SIFT base features and creates a DEG graph with them. The graph is than stored on the drive.
    // if(std::filesystem::exists(optimized_graph_file.c_str()) == false)
    //     rng_optimize_graph(graph_file, optimized_graph_file);

    // // loads the graph from the drive and test it against the SIFT query data
    // test_graph(data_path, optimized_graph_file, repeat_test, test_k);






    // const auto graph_file_out = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0_postOrdered232076720.test.deg").string();
    // reorder_graph(graph_file, order_file, graph_file_out);
    // naive_reorder_graph(graph_file, order_file,  graph_file_out);
    // test_graph(data_path, graph_file_out, repeat_test, test_k);

    fmt::print("Test OK\n");
    return 0;
}