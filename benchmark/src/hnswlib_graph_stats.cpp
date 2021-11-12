#include <omp.h>
#include <fmt/core.h>
#include <tsl/robin_set.h>

#include <algorithm>
#include <limits>

#include "hnswlib.h"
#include "hnsw/utils.h"
#include "stopwatch.h"

static auto read_top_list(const char* fname, size_t& d_out, size_t& n_out)
{
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{})
    {
        fmt::print(stderr, "error when accessing top list file {}, size is: {} message: {} \n", fname, file_size, ec.message());
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

    auto l2space = hnswlib::L2Space(128);
    auto graph = new hnswlib::HierarchicalNSW<float>(&l2space, graph_file, false);
    auto graph_size = graph->cur_element_count;
    auto max_edges_per_node = graph->size_links_level0_;

    size_t top_list_dims;
    size_t top_list_count;
    const auto all_top_list = read_top_list(top_list_file, top_list_dims, top_list_count);
    fmt::print("Load TopList from file {} with {} elements and k={}\n", top_list_file, top_list_count, top_list_dims);

    if(top_list_count != graph_size) {
        fmt::print(stderr, "The number of elements in the TopList file is different than in the graph: {} vs {}\n", top_list_count, graph_size);
        return;
    }

    if(top_list_dims < max_edges_per_node) {
        fmt::print(stderr, "Edges per node {} is higher than the TopList size = {} \n", max_edges_per_node, top_list_dims);
        return;
    }
    
    // compute the graph quality
    uint64_t perfect_neighbor_count = 0;
    uint64_t total_neighbor_count = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto linklist_data = graph->get_linklist_at_level(n, 0);
        auto edges_per_node = graph->getListCount(linklist_data);
        auto neighbor_indizies = (tableint*)(linklist_data + 1);

        // get top list of this node
        auto top_list = all_top_list.get() + n * top_list_dims;
        if(top_list_dims < edges_per_node) {
            fmt::print("TopList for {} is not long enough has {} elements has {}\n", n, edges_per_node, top_list_dims);
            edges_per_node = (uint16_t) top_list_dims;
        }
        total_neighbor_count += edges_per_node;

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
    auto perfect_neighbor_ratio = (float) perfect_neighbor_count / total_neighbor_count;
    auto avg_edge_count = (float) total_neighbor_count / graph_size;

    // compute the min, and max out degree
    uint16_t min_out = (uint16_t) max_edges_per_node;
    uint16_t max_out = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto linklist_data = graph->get_linklist_at_level(n, 0);
        auto edges_per_node = graph->getListCount(linklist_data);

        if(edges_per_node < min_out)
            min_out = edges_per_node;
        if(max_out < edges_per_node)
            max_out = edges_per_node;
    }

    // compute the min, and max in degree
    auto in_degree_count = std::vector<uint32_t>(graph_size);
    for (uint32_t n = 0; n < graph_size; n++) {
        auto linklist_data = graph->get_linklist_at_level(n, 0);
        auto edges_per_node = graph->getListCount(linklist_data);
        auto neighbor_indizies = (tableint*)(linklist_data + 1);

        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = neighbor_indizies[e];
            in_degree_count[neighbor_index]++;
        }
    }


    uint32_t min_in = std::numeric_limits<uint32_t>::max();
    uint32_t max_in = 0;
    uint32_t zero_in_count = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto in_degree = in_degree_count[n];

        if(in_degree < min_in)
            min_in = in_degree;
        if(max_in < in_degree)
            max_in = in_degree;

        if(in_degree == 0) {
            zero_in_count++;
            fmt::print("Node {} has zero incoming connections\n", n);
        }
    }

    fmt::print("GQ {}, avg degree {}, min_out {}, max_out {}, min_in {}, max_in {}, zero in nodes {}\n", perfect_neighbor_ratio, avg_edge_count, min_out, max_out, min_in, max_in, zero_in_count);
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
    const auto graph_file = (data_path / "hnsw" / "sift1m_ef_500_M_24.bin").string(); // GQ 0.36056197, avg degree 29.743711, min_out 1, max_out 48, min_in 0, max_in 182, zero in nodes 3

    compute_stats(graph_file.c_str(), top_list_file.c_str());

    fmt::print("Test OK\n");
    return 0;
}