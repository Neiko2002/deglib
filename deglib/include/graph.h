#pragma once

#include <assert.h>
#include <tsl/robin_hash.h>
#include <tsl/robin_map.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "repository.h"

namespace deglib
{

struct Neighbor
{
    uint32_t id;
    float distance;
    const float* feature_vector;
};

class Graph
{
  public:
    Graph(uint32_t nodes_count) : nodes_{nodes_count} {}

    size_t size() const { return nodes_.size(); }

    const auto& nodes() const { return nodes_; }

    auto& nodes() { return nodes_; }

    const auto& edges(const uint32_t nodeid) const { return nodes_.find(nodeid)->second; }

  private:
    tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, Neighbor>> nodes_;
};

/**
 * Load the graph
 **/
Graph load_graph(const char* path_graph, const deglib::FeatureRepository &repository)
{
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(path_graph, ec);
    if (ec != std::error_code{})
    {
        fmt::print(stderr, "error when accessing test file, size is: {} message: {} \n", file_size, ec.message());
        perror("");
        abort();
    }

    auto ifstream = std::ifstream(path_graph, std::ios::binary);
    if (!ifstream.is_open())
    {
        fmt::print(stderr, "could not open {}\n", path_graph);
        perror("");
        abort();
    }

    // read the entire file into a buffer
    auto buffer = std::make_unique<char[]>(file_size);
    if (!ifstream.read(buffer.get(), file_size))
    {
        fmt::print(stderr, "unable to read file content {}\n", path_graph);
        perror("");
        abort();
    }
    ifstream.close();

    // the file only contains ints and floats
    auto file_values = (uint32_t*)buffer.get();
    const uint32_t node_count = *(file_values++);
    auto graph = Graph(node_count);
    auto&& nodes = graph.nodes();
    for (uint32_t node_idx = 0; node_idx < node_count; node_idx++)
    {
        const auto node_id = *(file_values++);
        const auto edge_count = *(file_values++);

        auto edges = tsl::robin_map<uint32_t, Neighbor>(edge_count);
        for (uint32_t edge_idx = 0; edge_idx < edge_count; edge_idx++)
        {
            const auto neighbor_id = *(file_values++);
            const auto distance = *(float*)(file_values++);
            edges[neighbor_id] = {neighbor_id, distance, repository.getFeature(neighbor_id)};
        }

        nodes[node_id] = std::move(edges);
    }

    return graph;
}

}  // namespace deglib
