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

   
    const uint32_t repeat_test = 1;
    const auto data_path = std::filesystem::path(DATA_PATH);
    

    // SIFT1M
    const uint32_t k = 1000; 
    const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string(); 
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.test.deg").string();  // fast 28min
    const auto path_query_groundtruth = (data_path / "SIFT1M" / "sift_explore_ground_truth.ivecs").string();
    const auto path_entry_vertex = (data_path / "SIFT1M" / "sift_explore_entry_vertex.ivecs").string();
    
    // 2DGraph
    // const uint32_t k = 10; 
    // const auto graph_file = (data_path / "L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd100+0_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.remove_non_rng_edges.deg").string();  
    // const auto path_query_groundtruth = (data_path / "explore_gt.ivecs").string();
    // const auto path_entry_vertex = (data_path / "explore_entry_vertex.ivecs").string();

    // load graph
    fmt::print("Load graph {} \n", graph_file);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    //const auto graph = deglib::graph::load_sizebounded_graph(graph_file.c_str());

    // load starting vertex data
    size_t entry_vertex_dims;
    size_t entry_vertex_count;
    const auto entry_vertex_f = deglib::fvecs_read(path_entry_vertex.c_str(), entry_vertex_dims, entry_vertex_count);
    const auto entry_vertex = (uint32_t*)entry_vertex_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} entry vertex {} dimensions \n", entry_vertex_count, entry_vertex_dims);

    // load ground truth data (nearest neighbors of the starting vertices)
    size_t ground_truth_dims;
    size_t ground_truth_count;
    const auto ground_truth_f = deglib::fvecs_read(path_query_groundtruth.c_str(), ground_truth_dims, ground_truth_count);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", ground_truth_count, ground_truth_dims);

    // explore the graph
    deglib::benchmark::test_graph_explore(graph, (uint32_t) ground_truth_count, ground_truth, (uint32_t) ground_truth_dims, entry_vertex, (uint32_t) entry_vertex_dims, repeat_test, k);

    fmt::print("Test OK\n");
    return 0;
}