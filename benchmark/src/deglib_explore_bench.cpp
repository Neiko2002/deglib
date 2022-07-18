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

    const uint32_t k = 10; 
    const uint32_t repeat_test = 1;
    const auto data_path = std::filesystem::path(DATA_PATH);
    

    
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.test.deg").string();  // fast 28min
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.test_rng_optimized.deg").string();  
    //const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.rng_optimized.deg.non_rng_removed.deg").string();  

    

    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k30nns_128D_L2_AddK30Eps0.2High_ImproveK30-2StepEps0.02Low_Path10_Rnd2+2_realHighLows_improveNonPerfectEdges.deg").string();  // fast 1h 8min    
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k24nns_128D_L2_AddK24Eps0.2High_ImproveK24-0StepEps0.001Low_Path5_Rnd5+4_realHighLows_improveNonPerfectEdges_noLoopDetection.deg").string();  // fast 1h 1min
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k24nns_128D_L2_AddK24Eps0.2High_ImproveK24-2StepEps0.02Low_Path10_Rnd5+5_improve_non_perfect_new_edges.deg").string();  // best 3h 1min
    
    // const auto graph_file = (data_path / "L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd100+0_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.deg").string();  
    const auto graph_file = (data_path / "L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd100+0_improveNonRNGAndSecondHalfOfNonPerfectEdges_RNGAddMinimalSwapAtStep0.add_rng_opt.remove_non_rng_edges.deg").string();  

    fmt::print("Load graph {} \n", graph_file);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    //const auto graph = deglib::graph::load_sizebounded_graph(graph_file.c_str());

    // test the graph
    //const auto path_query_groundtruth = (data_path / "SIFT1M" / "sift_explore_ground_truth.ivecs").string();
     const auto path_query_groundtruth = (data_path / "explore_gt.ivecs").string();
    size_t ground_truth_dims;
    size_t ground_truth_count;
    const auto ground_truth_f = deglib::fvecs_read(path_query_groundtruth.c_str(), ground_truth_dims, ground_truth_count);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", ground_truth_count, ground_truth_dims);

    //const auto path_entry_node = (data_path / "SIFT1M" / "sift_explore_entry_node.ivecs").string();
     const auto path_entry_node = (data_path / "explore_entry_node.ivecs").string();
    size_t entry_node_dims;
    size_t entry_node_count;
    const auto entry_node_f = deglib::fvecs_read(path_entry_node.c_str(), entry_node_dims, entry_node_count);
    const auto entry_node = (uint32_t*)entry_node_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} entry node {} dimensions \n", entry_node_count, entry_node_dims);

    deglib::benchmark::test_graph_explore(graph, (uint32_t) ground_truth_count, ground_truth, (uint32_t) ground_truth_dims, entry_node, (uint32_t) entry_node_dims, repeat_test, k);

    fmt::print("Test OK\n");
    return 0;
}