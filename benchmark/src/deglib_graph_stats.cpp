#include <random>
#include <chrono>

#include <fmt/core.h>

#include "benchmark.h"
#include "deglib.h"

static void compute_stats(const char* graph_file, const char* top_list_file) {
    fmt::print("Compute graph stats of {}\n", graph_file);

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
    
    auto perfect_neighbor_count = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto neighbor_indizies = graph.getNeighborIndizies(n);
        auto top_list = all_top_list + n * top_list_dims;

        // check if every neighbor is from the perfect neighborhood
        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = neighbor_indizies[e];

            // find in the neighbor ini the first few elements of the top list
            for (uint32_t i = 0; i < edges_per_node; i++) {
                if(neighbor_index == top_list[i]) {
                    perfect_neighbor_count++;
                    break;
                }
            }
        }
    }

    auto perfect_neighbor_ratio = (float) perfect_neighbor_count / (graph_size * edges_per_node);
    fmt::print("Graph quality is {}\n", perfect_neighbor_ratio);
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
    const auto top_list_file = (data_path / "SIFT1M" / "sift_base_top1000.ivecs").string();

    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions" / "k30nns_128D_L2_AddK30Eps0.2.deg").string();  // GQ=0.47360423
    // const auto graph_file = (data_path / "deg" / "best_distortion_decisions/improve" / "deg30_128D_L2_AddK30Eps0.2_Improve30Eps0.02_ImproveExt30-2StepEps0.02_Path15_it1m.deg").string();  // GQ=0.50627613
    const auto graph_file = (data_path / "deg" / "best_distortion_decisions/improve" / "deg30_128D_L2_AddK30Eps0.2_Improve30Eps0.02_ImproveExt30-2StepEps0.02_Path15_it20m.deg").string();  // GQ=0.5304048
    
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

    compute_stats(graph_file.c_str(), top_list_file.c_str());

    fmt::print("Test OK\n");
    return 0;
}