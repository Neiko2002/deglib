#include <random>
#include <chrono>

#include <fmt/core.h>

#include "benchmark.h"
#include "deglib.h"

static void compute_graph_quality(const char* graph_file, const char* top_list_file) {
    fmt::print("Compute graph quality of {}\n", graph_file);

    auto graph = deglib::graph::load_sizebounded_graph(graph_file);
    const auto graph_size = graph.size();
    const auto edges_per_node = graph.getEdgesPerNode();

    size_t top_list_dims;
    size_t top_list_count;
    const auto ground_truth_f = deglib::fvecs_read(top_list_file, top_list_dims, top_list_count);
    const auto all_top_list = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("Load TopList from file {} with {} elements and k={}\n", top_list_file, top_list_count, top_list_dims);

    if(top_list_count != graph_size) {
        fmt::print(stderr, "The number of elements in the TopList file is different than in the graph: {} vs {}\n", top_list_count, graph_size);
        return;
    }

    if(top_list_dims < edges_per_node) {
        fmt::print(stderr, "Edges per node {} is higher than the TopList size = {} \n", edges_per_node, top_list_dims);
        return;
    }
    
    uint64_t perfect_neighbor_count = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto neighbor_indices = graph.getNeighborIndices(n);
        auto top_list = all_top_list + n * top_list_dims;

        // check if every neighbor is from the perfect neighborhood
         fmt::print("Neighbors of vertex {}\n", n);
        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = neighbor_indices[e];
            fmt::print("{} = {}\n", e, neighbor_indices[e]);

            // find in the neighbor in the first few elements of the top list
            for (uint32_t i = 0; i < edges_per_node; i++) {
                if(neighbor_index == top_list[i]) {
                    perfect_neighbor_count++;
                    break;
                }
            }
        }
        fmt::print("\n");
    }

    auto perfect_neighbor_ratio = (float) perfect_neighbor_count / (graph_size * edges_per_node);
    auto connected = deglib::analysis::check_graph_connectivity(graph);
    fmt::print("Graph quality is {}, edges per node {}, graph size {}, connected {}\n", perfect_neighbor_ratio, edges_per_node, graph_size, connected);
}

static void compute_edge_histogram(const char* graph_file, const uint32_t steps, const float interval) {
    fmt::print("Analyse edge weights of {}\n", graph_file);

    auto graph = deglib::graph::load_sizebounded_graph(graph_file);
    const auto graph_size = graph.size();
    const auto edges_per_node = graph.getEdgesPerNode();

    fmt::print("Compute max and min weight\n");
    {
        float min_weight = std::numeric_limits<float>::max();
        float max_weight = std::numeric_limits<float>::min();
        for (uint32_t n = 0; n < graph_size; n++) {
            const auto edge_weights = graph.getNeighborWeights(n);
            for (size_t e = 0; e < edges_per_node; e++) {
                if(edge_weights[e] < min_weight)
                    min_weight = edge_weights[e];
                if(edge_weights[e] > max_weight)
                    max_weight = edge_weights[e];
            }
        }
        fmt::print("max weight {}, min weight {} \n", max_weight, min_weight);
    }

    fmt::print("Histogram of all edge weights\n");
    {
        auto edge_weight_histogram = std::vector<uint32_t>(steps);
        for (uint32_t n = 0; n < graph_size; n++) {
            const auto edge_weights = graph.getNeighborWeights(n);
            for (size_t e = 0; e < edges_per_node; e++) {
                const auto bin = std::min(steps-1, uint32_t(edge_weights[e] / interval));
                edge_weight_histogram[bin]++;
            }
        }

        for (size_t i = 0; i < steps; i++) 
            fmt::print("{}-{} {}\n", i*interval, (i+1)*interval, edge_weight_histogram[i]);
    }
    
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

    const auto data_path = std::filesystem::path(DATA_PATH);
    // const auto top_list_file = (data_path / "SIFT1M" / "sift_base_top1000.ivecs").string();
    // const auto top_list_file  = (data_path / "glove-100" / "glove_base_top1000.ivecs").string(); 
    const auto top_list_file  = (data_path / "base_top13.ivecs").string(); 
    auto graph_files = std::vector<std::string>();

    // 2D Graph
    graph_files.emplace_back((data_path / "L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.deg").string()); 
    graph_files.emplace_back((data_path / "L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd10+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.deg").string()); 


    // SIFT1M
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k30nns_128D_L2_AddK30Eps0.2.deg").string();  // GQ=0.47360423
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions/improve" / "deg30_128D_L2_AddK30Eps0.2_Improve30Eps0.02_ImproveExt30-2StepEps0.02_Path15_it1m.deg").string();  // GQ=0.50627613
    //const auto graph_file = (data_path / "deg" / "best_distortion_decisions/improve" / "deg30_128D_L2_AddK30Eps0.2_Improve30Eps0.02_ImproveExt30-2StepEps0.02_Path15_it20m.deg").string();  // GQ=0.5304048

    // graph_files.emplace_back((data_path / "deg" / "best_distortion_decisions" / "k30nns_128D_L2_AddK30Eps0.2Low.deg").string());
    // graph_files.emplace_back((data_path / "deg" / "best_distortion_decisions" / "k30nns_128D_L2_AddK30Eps0.2High_ImproveK30-2StepEps0.02Low_Path10_Rnd2+2_realHighLows_improveNonPerfectEdges.deg").string());
    // graph_files.emplace_back((data_path / "deg" / "best_distortion_decisions" / "k30nns_128D_L2_AddK30Eps0.2High_ImproveK30-2StepEps0.02Low_Path10_Rnd5+5_realHighLows_improveNonPerfectEdges.deg").string());

    // graph_files.emplace_back((data_path / "deg" / "paper" / "k24nns_128D_L2_AddK24Eps0.2High_ImproveK24-2StepEps0.02Low_Path10_Rnd5+5.deg").string());  
    // graph_files.emplace_back((data_path / "deg" / "paper" / "k24nns_128D_L2_AddK24Eps0.2High_ImproveK24-2StepEps0.02Low_Path10_Rnd0+0.deg").string());  

    // graph_files.emplace_back((data_path / "deg" / "best_distortion_decisions" / "k24nns_128D_L2_AddK24Eps0.2High_ImproveK24-2StepEps0.02Low_Path10_Rnd15+15.deg").string());  
    // graph_files.emplace_back((data_path / "deg" / "best_distortion_decisions" / "k24nns_128D_L2_AddK24Eps0.2High_ImproveK24-2StepEps0.02Low_Path10_Rnd1+1-rerun.deg").string());  

    // graph_files.emplace_back((data_path / "deg" / "best_distortion_decisions" / "k24nns_128D_L2_AddK24Eps0.2High_ImproveK24Eps0.02Low_ImproveExtK24-2StepEps0.02Low_Path10_Rnd1+1.deg").string());  
    // graph_files.emplace_back((data_path / "deg" / "best_distortion_decisions" / "k24nns_128D_L2_AddK24Eps0.2High_ImproveK24Eps0.02Low_ImproveExtK24-2StepEps0.02Low_Path10_Rnd5+5.deg").string());  






    // GloVe
    // graph_files.emplace_back((data_path / "deg" / "k30nns_100D_L2_AddK30Eps0.2High.deg").string());
    // graph_files.emplace_back((data_path / "deg" / "k30nns_100D_L2_AddK30Eps0.2High_ImproveK30-2StepEps0.02Low_Path10_Rnd0+0_improveNonPerfectEdges.deg").string());
    // graph_files.emplace_back((data_path / "deg" / "k30nns_100D_L2_AddK30Eps0.2High_ImproveK30-2StepEps0.02Low_Path10_Rnd0+0_realHighLow_improveNonPerfectEdges.deg").string());
    // graph_files.emplace_back((data_path / "deg" / "k30nns_100D_L2_AddK30Eps0.2High_ImproveK30-2StepEps0.02Low_Path10_Rnd3+3_realHighLow.deg").string());
    // graph_files.emplace_back((data_path / "deg" / "k30nns_100D_L2_AddK30Eps0.2High_ImproveK30-2StepEps0.02Low_Path10_Rnd3+3_realHighLow-rerun.deg").string());

    // // Graph quality is 0.23271069, edges per node 30, graph size 1183514, connected true
    // graph_files.emplace_back((data_path / "deg" / "100D_L2_K30_AddK30Eps0.2High_ImproveK30-0StepEps0.001Low_Path5_Rnd10+10_realHighLow_improveTheBetterHalfOfTheNonPerfectEdges.deg").string());

    // // Graph quality is 0.23340161, edges per node 30, graph size 1183514, connected true
    // graph_files.emplace_back((data_path / "deg" / "100D_L2_K30_AddK30Eps0.2High_ImproveK30-0StepEps0.001Low_Path5_Rnd10+10_realHighLow_improveTheBetterHalfOfTheNonPerfectEdgesWithHighHighImprov.deg").string());

    // // Graph quality is 0.23545972, edges per node 30, graph size 1183514, connected true
    // graph_files.emplace_back((data_path / "deg" / "100D_L2_K30_AddK30Eps0.3High_ImproveK30-0StepEps0.001Low_Path5_Rnd10+10_realHighLow_improveTheBetterHalfOfTheNonPerfectEdgesWithHighHighImprov.deg").string()); // 30 best

    // // Graph quality is 0.23538786, edges per node 30, graph size 1183514, connected true
    // graph_files.emplace_back((data_path / "deg" / "100D_L2_K30_AddK30Eps0.3High_ImproveK30-0StepEps0.001Low_Path5_Rnd10+10_realHighLow_improveTheBetterHalfOfTheNonPerfectEdgesWithHighHighImprov_noLoopDetection.deg").string());

    // // Graph quality is 0.23036574, edges per node 60, graph size 1183514, connected true
    // graph_files.emplace_back((data_path / "deg" / "k60nns_100D_L2_AddK60Eps0.2High_ImproveK60Eps0.02Low_ImproveExtK60-3StepEps0.02Low_Path10_Rnd2+2.deg").string()); // 60 best


    // graph_files.emplace_back((data_path / "deg" / "k60nns_100D_L2_AddK60Eps0.05High_ImproveK60-2StepEps0.001Low_Path10_Rnd0+0_realHighLow_improveNonPerfectEdges.deg").string());
    // graph_files.emplace_back((data_path / "deg" / "k60nns_100D_L2_AddK60Eps0.05High_ImproveK60-2StepEps0.001Low_Path10_Rnd3+3_realHighLow.deg").string());
    // graph_files.emplace_back((data_path / "deg" / "k60nns_100D_L2_AddK60Eps0.2High.deg").string());


    
    





    
    // graph_files.emplace_back((data_path / "deg" / "k30nns_100D_L2_AddK30Eps0.2High_ImproveK30Eps0.02Low_ImproveExtK30-2StepEps0.02Low_Path10_Rnd4+4.deg").string());    // HighLowLow
    // graph_files.emplace_back((data_path / "deg" / "k30nns_100D_L2_AddK30Eps0.2High_ImproveK30Eps0.02High_ImproveExtK30-2StepEps0.02High_Path10_Rnd2+2.deg").string());  // HighHighHigh
    // graph_files.emplace_back((data_path / "deg" / "k30nns_100D_L2_AddK30Eps0.2Low_ImproveK30Eps0.02Low_ImproveExtK30-2StepEps0.02Low_Path20_Rnd10+3.deg").string());    // LowLowLow
    // graph_files.emplace_back((data_path / "deg" / "k30nns_100D_L2_AddK30Eps0.2Low_ImproveK30Eps0.02High_ImproveExtK30-2StepEps0.02High_Path10_Rnd3+3.deg").string());   // LowHighHigh
    // graph_files.emplace_back((data_path / "deg" / "k30nns_100D_L2_AddK30Eps0.2High.deg").string());                                                                     // High
    // graph_files.emplace_back((data_path / "deg" / "k30nns_100D_L2_AddK30Eps0.2Low.deg").string());                                                                      // Low

    
    
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k30nns_128D_L2_AddK30Eps0.2_ImproveK30Eps0.02_ImproveExtK30-2StepEps0.02_Path20_Rnd15+15.deg").string();  // GQ=0.5285193
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k30nns_128D_L2_AddK30Eps0.2_ImproveK30Eps0.02_ImproveExtK30-2StepEps0.02_Path20_Rnd15+15-rerun.deg").string();  // GQ=0.52845836
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k30nns_128D_L2_AddK30Eps0.2_ImproveK30Eps0.02_ImproveExtK30-2StepEps0.02_Path20_Rnd15+15-rerun2.deg").string();  // GQ=0.5281604


    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k40nns_128D_L2_AddK40Eps0.1_ImproveK40Eps0.02_ImproveExtK40-2StepEps0.02_Path12_Rnd5+5.deg").string();  // GQ=0.5232597
    //const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k24nns_128D_L2_AddK24Eps0.2_ImproveK24Eps0.02_ImproveExtK24-2StepEps0.02_Path10_Rnd5+5.deg").string();  // GQ=0.51845485
    //const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k24nns_128D_L2_AddK24Eps0.2_ImproveK24Eps0.02_ImproveExtK24-2StepEps0.02_Path10_Rnd15+15.deg").string();  // GQ=0.52803755

    //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2_ImproveK24Eps0.02_ImproveExtK36-2StepEps0.02.deg").string(); // best with GQ=0.48901713
    //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK20Eps0.2_ImproveK20Eps0.02_ImproveExtK12-2StepEps0.02.deg").string(); // fast with GQ=0.51287943
    //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2_ImproveK24Eps0.02.deg").string(); // simple improve only with GQ=0.48969483
    //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2.deg").string(); // build only with GQ=0.4492762

    for (auto &&graph_file : graph_files) {
        compute_graph_quality(graph_file.c_str(), top_list_file.c_str());
        // compute_edge_histogram(graph_file.c_str(), 50, 10000); // SIFT
        // compute_edge_histogram(graph_file.c_str(), 50, 4); // GloVe
    }

    fmt::print("Test OK\n");
    return 0;
}