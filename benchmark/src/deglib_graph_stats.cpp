#include <random>
#include <chrono>

#include <fmt/core.h>

#include "benchmark.h"
#include "deglib.h"

static void compute_graph_quality(const char* graph_file, const char* top_list_file) {
    fmt::print("Compute graph quality of {}\n", graph_file);

    auto graph = deglib::graph::load_sizebounded_graph(graph_file);
    const auto graph_size = graph.size();
    const auto edges_per_vertex = graph.getEdgesPerNode();

    size_t top_list_dims;
    size_t top_list_count;
    const auto ground_truth_f = deglib::fvecs_read(top_list_file, top_list_dims, top_list_count);
    const auto all_top_list = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("Load TopList from file {} with {} elements and k={}\n", top_list_file, top_list_count, top_list_dims);

    if(top_list_count != graph_size) {
        fmt::print(stderr, "The number of elements in the TopList file is different than in the graph: {} vs {}\n", top_list_count, graph_size);
        return;
    }

    if(top_list_dims < edges_per_vertex) {
        fmt::print(stderr, "Edges per vertex {} is higher than the TopList size = {} \n", edges_per_vertex, top_list_dims);
        return;
    }
    
    uint64_t perfect_neighbor_count = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto neighbor_indices = graph.getNeighborIndices(n);
        auto top_list = all_top_list + n * top_list_dims;

        // check if every neighbor is from the perfect neighborhood
        //  fmt::print("Neighbors of vertex {}\n", n);
        for (uint32_t e = 0; e < edges_per_vertex; e++) {
            auto neighbor_index = neighbor_indices[e];
            // fmt::print("{} = {}\n", e, neighbor_indices[e]);

            // find in the neighbor in the first few elements of the top list
            for (uint32_t i = 0; i < edges_per_vertex; i++) {
                if(neighbor_index == top_list[i]) {
                    perfect_neighbor_count++;
                    break;
                }
            }
        }
        // fmt::print("\n");
    }

    auto perfect_neighbor_ratio = (float) perfect_neighbor_count / (graph_size * edges_per_vertex);
    auto connected = deglib::analysis::check_graph_connectivity(graph);
    fmt::print("Graph quality is {}, edges per vertex {}, graph size {}, connected {}\n", perfect_neighbor_ratio, edges_per_vertex, graph_size, connected);
}

static void compute_edge_histogram(const char* graph_file, const uint32_t steps, const float interval) {
    fmt::print("Analyse edge weights of {}\n", graph_file);

    auto graph = deglib::graph::load_sizebounded_graph(graph_file);
    const auto graph_size = graph.size();
    const auto edges_per_vertex = graph.getEdgesPerNode();

    fmt::print("Compute max and min weight\n");
    {
        float min_weight = std::numeric_limits<float>::max();
        float max_weight = std::numeric_limits<float>::min();
        for (uint32_t n = 0; n < graph_size; n++) {
            const auto edge_weights = graph.getNeighborWeights(n);
            for (size_t e = 0; e < edges_per_vertex; e++) {
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
            for (size_t e = 0; e < edges_per_vertex; e++) {
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
    auto graph_files = std::vector<std::string>();

    // ------------------------------------- 2D Graph -------------------------------------------
    // graph_files.emplace_back((data_path / "L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd0+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.deg").string()); 
    // graph_files.emplace_back((data_path / "L2_K4_AddK10Eps0.2High_SwapK10-0StepEps0.001LowPath5Rnd10+0_improveTheBetterHalfOfTheNonPerfectEdges_RNGAddMinimalSwapAtStep0.deg").string()); 
    // const auto top_list_file  = (data_path / "base_top13.ivecs").string(); 

    // ------------------------------------- SIFT1M -------------------------------------------
    // graph_files.emplace_back((data_path / "deg" / "best_distortion_decisions" / "128D_L2_K30_AddK60Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string()); // GQ 0.49960038
    // const auto top_list_file = (data_path / "SIFT1M" / "sift_base_top1000.ivecs").string();

    // ------------------------------------- Glove -------------------------------------------
    // graph_files.emplace_back((data_path / "deg" / "100D_L2_K30_AddK30Eps0.2High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string()); // GQ 0.21667433
    // const auto top_list_file  = (data_path / "glove-100" / "glove-100_base_top1000.ivecs").string(); 

    // ------------------------------------- Enron -------------------------------------------
    // graph_files.emplace_back((data_path / "deg" / "1369D_L2_K30_AddK60Eps0.3High_SwapK30-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string()); // GQ 0.36472255
    // const auto top_list_file  = (data_path / "enron" / "enron_base_top1000.ivecs").string(); 

    // ------------------------------------- Audio -------------------------------------------
    graph_files.emplace_back((data_path / "deg" / "192D_L2_K20_AddK40Eps0.3High_SwapK20-0StepEps0.001LowPath5Rnd0+0_improveEvery2ndNonPerfectEdge.deg").string()); // GQ 0.36472255
    const auto top_list_file  = (data_path / "audio" / "audio_base_top1000.ivecs").string(); 

    for (auto &&graph_file : graph_files) {
        compute_graph_quality(graph_file.c_str(), top_list_file.c_str());
        // compute_edge_histogram(graph_file.c_str(), 50, 10000); // SIFT
        // compute_edge_histogram(graph_file.c_str(), 50, 4); // GloVe
    }

    fmt::print("Test OK\n");
    return 0;
}