#pragma once

#include "repository.h"

#include <assert.h>
#include <tsl/robin_hash.h>
#include <tsl/robin_map.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace deglib
{
struct Neighbor
{
    uint32_t id;
    float distance;
    const float* feature_vector;

    // Neighbor(uint32_t neighbor_id, float neighbor_distance, float* neighbor_feature_vector) : id(neighbor_id),
    // distance(neighbor_distance), feature_vector(neighbor_feature_vector) {}
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
Graph load_graph(const char* path_graph, const deglib::FeatureRepository& repository)
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

template <size_t Dimensions>
struct Node
{
    uint32_t id;
    // float distance;
    // uin32_t size;
    std::array<float, Dimensions> feature_vector;
};

template <size_t EdgeCount, size_t Dimensions>
struct Edges
{
    // uin32_t size;
    std::array<deglib::Node<Dimensions>, EdgeCount> values;
};

class StaticGraph
{
  public:
    StaticGraph(std::shared_ptr<void> nodes, size_t size_, size_t edge_count_, size_t dims_)
        : nodes_(std::move(nodes)), size_(size_), edge_count_(edge_count_), dims_(dims_)
    {
    }

    auto size() const { return size_; }

    auto edge_count() const { return edge_count_; }

    auto dims() const { return dims_; }

    const auto& nodes() const { return nodes_; }

    auto& nodes() { return nodes_; }

    template <size_t EdgeCount, size_t Dimensions>
    const auto& edges(const uint32_t nodeid) const
    {
        return static_cast<Edges<EdgeCount, Dimensions>*>(nodes_.get())[nodeid];
    }

  private:
    std::shared_ptr<void> nodes_;
    size_t size_;
    size_t edge_count_;
    size_t dims_;
};

struct GraphConstructorFunctor
{
    uint32_t* file_values;
    size_t node_count;
    const deglib::FeatureRepository& repository;

    template <size_t EdgeCount, size_t Dimensions>
    auto operator()()
    {
        auto nodes = std::make_unique<deglib::Edges<EdgeCount, Dimensions>[]>(node_count);
        /*node_count*/ ++file_values;
        for (uint32_t node_idx = 0; node_idx < node_count; node_idx++)
        {
            const auto node_id = *(file_values++);
            /*edge_count*/ file_values++;
            for (auto&& edge : nodes[node_id].values)
            {
                // sort edges by distance
                const auto neighbor_id = *(file_values++);
                /*distance*/ *reinterpret_cast<float*>(file_values++);
                edge.id = neighbor_id;
                const auto* features = repository.getFeature(neighbor_id);
                std::copy_n(features, Dimensions, edge.feature_vector.begin());
            }
        }
        return deglib::StaticGraph(std::move(nodes), node_count, EdgeCount, Dimensions);
    }
};

template <size_t EdgeCount, class Function>
auto invoke_chosing_feature_vector_size(Function&& function, size_t feature_vector_size)
{
    if (feature_vector_size != 128)
    {
        assert(!"unimplemented");
        std::abort();
    }
    return function.template operator()<EdgeCount, 128>();
}

template <class Function>
auto invoke_chosing_edge_count_and_feature_vector_size(Function&& function, size_t edge_count,
                                                       size_t feature_vector_size)
{
    if (edge_count != 24)
    {
        assert(!"unimplemented");
        std::abort();
    }
    return invoke_chosing_feature_vector_size<24>(function, feature_vector_size);
}

/**
 * Load the graph
 **/
StaticGraph load_static_graph(const char* path_graph, const deglib::FeatureRepository& repository)
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
    if (!ifstream.read(reinterpret_cast<char*>(buffer.get()), file_size))
    {
        fmt::print(stderr, "unable to read file content {}\n", path_graph);
        perror("");
        abort();
    }
    ifstream.close();

    // the file only contains ints and floats
    auto file_values = (uint32_t*)buffer.get();
    const auto node_count = *(file_values);
    const auto edge_count = *(file_values + 2);
    // assuming all edges have the same size

    static constexpr auto FEATURE_VECTOR_SIZE = 128;
    auto graph = invoke_chosing_edge_count_and_feature_vector_size(
        GraphConstructorFunctor{file_values, node_count, repository}, edge_count, FEATURE_VECTOR_SIZE);

    return graph;
}

}  // namespace deglib
