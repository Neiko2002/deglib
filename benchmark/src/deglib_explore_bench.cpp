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

    const uint32_t k = 500; 
    const uint32_t repeat_test = 1;
    const auto data_path = std::filesystem::path(DATA_PATH);

    //const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "add_only" / "k60nns_128D_L2_AddK60Eps0.2.deg").string(); 
    //const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k40nns_128D_L2_AddK40Eps0.1_ImproveK40Eps0.02_ImproveExtK40-2StepEps0.02_Path12_Rnd5+5.deg").string();    // best 7h 29min

    //const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "add_only" / "k30nns_128D_L2_AddK30Eps0.2.deg").string();                                                    // add only 9min
    //const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k30nns_128D_L2_AddK30Eps0.2_ImproveK30Eps0.02_ImproveExtK30-2StepEps0.02_Path20_Rnd10+10.deg").string();  // best 10h 46min

    //const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k24nns_128D_L2_AddK24Eps0.2.deg").string();                                                               // add only 8min
    //const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k24nns_128D_L2_AddK24Eps0.2_ImproveK24Eps0.02_ImproveExtK24-2StepEps0.02_Path10_Rnd5+5.deg").string();    // fast 1h 43min
    const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k24nns_128D_L2_AddK24Eps0.2_ImproveK24Eps0.02_ImproveExtK24-2StepEps0.02_Path10_Rnd15+15.deg").string();  // best 6h 31min

    // load an existing graph
    //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2.deg").string();                                               // add only 8min
    //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2_ImproveK20Eps0.02.deg").string();                             // simple improve 24min
    //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2_ImproveK20Eps0.02_ImproveExtK14-2StepEps0.01.deg").string();  // fast 1h 21min
    //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2_ImproveK24Eps0.02_ImproveExtK36-2StepEps0.02.deg").string();    // best 8h 15min
    fmt::print("Load graph {} \n", graph_file);
    //const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    const auto graph = deglib::graph::load_sizebounded_graph(graph_file.c_str());

    // test the graph
    const auto path_query_groundtruth = (data_path / "SIFT1M" / "sift_explore_ground_truth.ivecs").string();
    size_t ground_truth_dims;
    size_t ground_truth_count;
    const auto ground_truth_f = deglib::fvecs_read(path_query_groundtruth.c_str(), ground_truth_dims, ground_truth_count);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", ground_truth_count, ground_truth_dims);

    const auto path_entry_node = (data_path / "SIFT1M" / "sift_explore_entry_node.ivecs").string();
    size_t entry_node_dims;
    size_t entry_node_count;
    const auto entry_node_f = deglib::fvecs_read(path_entry_node.c_str(), entry_node_dims, entry_node_count);
    const auto entry_node = (uint32_t*)entry_node_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} entry node {} dimensions \n", entry_node_count, entry_node_dims);

    deglib::benchmark::test_graph_explore(graph, (uint32_t) ground_truth_count, ground_truth, (uint32_t) ground_truth_dims, entry_node, (uint32_t) entry_node_dims, repeat_test, k);

    fmt::print("Test OK\n");
    return 0;
}