#include <random>
#include <chrono>

#include <fmt/core.h>

#include "benchmark.h"
#include "deglib.h"


static auto read_top_list(const char* fname, size_t& d_out, size_t& n_out)
{
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{})
    {
        fmt::print(stderr, "error when accessing top list file, size is: {} message: {} \n", fname, file_size, ec.message());
        perror("");
        abort();
    }

    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open())
    {
        fmt::print(stderr, "could not open {}\n", fname);
        perror("");
        abort();
    }

    uint32_t dims;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(int));
    assert((dims > 0 && dims < 1000000) || !"unreasonable dimension");
    assert((file_size - 4) % ((dims + 1) * 4) == 0 || !"weird file size");
    size_t n = (file_size - 4) / ((dims + 1) * 4);

    d_out = dims;
    n_out = n;

    auto x = std::make_unique<uint32_t[]>(n * (dims + 1));
    ifstream.read(reinterpret_cast<char*>(x.get()), n * (dims + 1) * sizeof(uint32_t));
    if (!ifstream) assert(ifstream.gcount() == static_cast<int>(n * (dims + 1)) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) memmove(&x[i * dims], &x[1 + i * (dims + 1)], dims * sizeof(uint32_t));

    ifstream.close();
    return x;
}

static void compute_stats(const char* graph_file, const char* top_list_file) {
    fmt::print("Compute graph stats of {}\n", graph_file);

    auto graph = deglib::graph::load_sizebounded_graph(graph_file);
    const auto graph_size = graph.size();
    const auto edges_per_node = graph.getEdgesPerNode();

    size_t top_list_dims;
    size_t top_list_count;
    const auto all_top_list = read_top_list(top_list_file, top_list_dims, top_list_count);
    fmt::print("Load TopList from file {} with {} elements and k={}\n", top_list_file, top_list_count, top_list_dims);

    if(top_list_count != graph_size) {
        fmt::print(stderr, "The number of elements in the TopList file is different than in the graph: {} vs {}\n", top_list_count, graph_size);
        return false;
    }

    if(top_list_dims < edges_per_node) {
        fmt::print(stderr, "Edges per node {} is higher than the TopList size = {} \n", edges_per_node, top_list_dims);
        return false;
    }
    
    auto perfect_neighbor_count = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto neighbor_indizies = graph.getNeighborIndizies(n);
        auto top_list = all_top_list.get() + n * top_list_dims;

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
    const auto top_list_file = (data_path / "SIFT1M/sift_base_top200_p0.998.ivecs").string();
    //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2_ImproveK24Eps0.02_ImproveExtK36-2StepEps0.02.deg").string(); // best with GQ=0.48901713
    //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK20Eps0.2_ImproveK20Eps0.02_ImproveExtK12-2StepEps0.02.deg").string(); // fast with GQ=0.51287943
    const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2_ImproveK24Eps0.02.deg").string(); // simple improve only with GQ=0.48969483
    //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2.deg").string(); // build only with GQ=0.4492762

    compute_stats(graph_file.c_str(), top_list_file.c_str());

    fmt::print("Test OK\n");
    return 0;
}