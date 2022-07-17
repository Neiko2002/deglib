#pragma once

#include <fmt/core.h>

#include "search.h"
#include "graph.h"

namespace deglib::analysis
{
    /**
     * Check if the number of nodes and edges is consistent. 
     * The edges of a node should only contain unique neighbor indices in ascending order and not a self-loop.
     * 
     * @param check_back_link checks if all edges are undirected (quite expensive)
     */
    static bool check_graph_validation(const deglib::search::SearchGraph& graph, const uint32_t expected_nodes, const bool check_back_link = false) {

        // check node count
        auto node_count = graph.size();
        if(node_count != expected_nodes) {
            fmt::print(stderr, "the graph has an unexpected number of nodes. expected {} got {} \n", expected_nodes, node_count);
            return false;
        }

        // check edges
        auto edges_per_node = graph.getEdgesPerNode();
        for (uint32_t n = 0; n < node_count; n++) {
            auto neighbor_indices = graph.getNeighborIndices(n);

            // check if the neighbor indizizes of the nodes are in ascending order and unique
            int64_t last_index = -1;
            for (size_t e = 0; e < edges_per_node; e++) {
                auto neighbor_index = neighbor_indices[e];

                if(n == neighbor_index) {
                    fmt::print(stderr, "node {} has a self-loop at position {} \n", n, e);
                    return false;
                }

                if(last_index == neighbor_index) {
                    fmt::print(stderr, "node {} has a duplicate neighbor at position {} with the neighbor index {} \n", n, e, neighbor_index);
                    return false;
                }

                if(last_index > neighbor_index) {
                    fmt::print(stderr, "the neighbor order for node {} is invalid: pos {} has index {} while pos {} has index {} \n", n, e-1, last_index, e, neighbor_index);
                    return false;
                }

                if(check_back_link && graph.hasEdge(neighbor_index, n) == false) {
                    fmt::print(stderr, "the neighbor {} of node {} does not have a back link to the node \n", neighbor_index, n);
                    return false;
                }

                last_index = neighbor_index;
            }
        }
        
        return true;
    }

    /**
     * Compute the graph quality be
     */
    static float calc_avg_edge_weight(const deglib::graph::MutableGraph& graph) {
        double total_distance = 0;
        uint64_t count = 0;

        const auto edges_per_node = graph.getEdgesPerNode();
        const auto node_count = graph.size();
        for (uint32_t n = 0; n < node_count; n++) {
            const auto weights = graph.getNeighborWeights(n);
            for (size_t e = 0; e < edges_per_node; e++)
                total_distance += weights[e];
            count += edges_per_node;
        }
        
        total_distance /= count;
        return (float) total_distance;
    }

    static auto calc_edge_weight_histogram(const deglib::graph::MutableGraph& graph, const bool sorted) {
 
        const auto edges_per_node = graph.getEdgesPerNode();
        const auto node_count = graph.size();
        auto all_edge_weights = std::vector<float>();
        all_edge_weights.reserve(edges_per_node*node_count);
        for (uint32_t n = 0; n < node_count; n++) {
            const auto weights = graph.getNeighborWeights(n);
            for (size_t e = 0; e < edges_per_node; e++)
                all_edge_weights.push_back(weights[e]);
        }

        if(sorted)
            std::sort(all_edge_weights.begin(), all_edge_weights.end());

        auto bin_count = 10;
        auto bin_size = all_edge_weights.size() / bin_count;
        auto avg_edge_weights = std::vector<float>(10);
        for (size_t bin = 0; bin < bin_count; bin++) {
            float weight_sum = 0;
            for (size_t n = 0; n < bin_size; n++) 
                weight_sum += all_edge_weights[bin_size * bin + n];
            avg_edge_weights[bin] = weight_sum / bin_size;
        }
        
        return avg_edge_weights;
    }

    /**
     * Check if the weights of the graph are still the same to the distance of the nodes
     */
    static float check_graph_weights(const deglib::graph::MutableGraph& graph) {
        const auto& feature_space = graph.getFeatureSpace();
        const auto dist_func = feature_space.get_dist_func();
        const auto dist_func_param = feature_space.get_dist_func_param();
        const auto edges_per_node = graph.getEdgesPerNode();
        const auto node_count = graph.size();

        for (uint32_t n = 0; n < node_count; n++) {
            const auto fv1 = graph.getFeatureVector(n);
            const auto neighborIds = graph.getNeighborIndices(n); 
            const auto neighborWeights = graph.getNeighborWeights(n); 
            for (uint32_t e = 0; e < edges_per_node; e++) {
                const auto fv2 = graph.getFeatureVector(neighborIds[e]);
                const auto dist = dist_func(fv1, fv2, dist_func_param);

                if(neighborWeights[e] != dist) {
                    fmt::print(stderr, "Node {} at edge index {} has a weight of {} to node {} but its distance is {} \n", n, e, neighborWeights[e], neighborIds[e], dist);
                    return false;
                }
            }
        }

        return true;
    }

    /**
     * Is the vertex_index a RNG conform neighbor if it gets connected to target_index?
     * 
     * Does vertex_index has a neighbor which is connected to the target_index and has a lower weight?
     */
    static auto checkRNG(const deglib::graph::MutableGraph& graph, const uint32_t edges_per_node, const uint32_t vertex_index, const uint32_t target_index, const float vertex_target_weight) {
      const auto neighbor_indices = graph.getNeighborIndices(vertex_index);
      const auto neighbor_weight = graph.getNeighborWeights(vertex_index);
      for (size_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {
        const auto neighbor_target_weight = graph.getEdgeWeight(neighbor_indices[edge_idx], target_index);  
        if(neighbor_target_weight >= 0 && vertex_target_weight > std::max(neighbor_weight[edge_idx], neighbor_target_weight)) {
          return false;
        }
      }
      return true;
    }

    /**
     * Is the vertex_index a RNG conform neighbor if it gets connected to target_index?
     */
    static auto checkRNG(const deglib::graph::MutableGraph& graph, const uint32_t vertex_index, const uint32_t target_index, const std::vector<std::pair<uint32_t, float>>& new_neighbors) {
      const auto dist_func = graph.getFeatureSpace().get_dist_func();
      const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
      const float vertex_target_weight = dist_func(graph.getFeatureVector(vertex_index), graph.getFeatureVector(target_index), dist_func_param); 

      const auto neighbor_size = new_neighbors.size();
      for (size_t n = 0; n < neighbor_size; n++) {
        const auto& new_neighbor = new_neighbors[n];
        const auto neighbor_target_weight = graph.getEdgeWeight(new_neighbor.first, target_index);
        if(neighbor_target_weight >= 0 && vertex_target_weight > std::max(new_neighbor.second, neighbor_target_weight)) {
         return false;
        }
      }
      return true;
    }

    static uint32_t calc_non_rng_edges(const deglib::graph::MutableGraph& graph) {
        const auto node_count = graph.size();
        const auto edge_per_node =graph.getEdgesPerNode();

        uint32_t removed_rng_edges = 0;
        for (uint32_t i = 0; i < node_count; i++) {
            const auto vertex_index = i;

            const auto neighbor_indices = graph.getNeighborIndices(vertex_index);
            const auto neighbor_weights = graph.getNeighborWeights(vertex_index);

            // find all none rng conform neighbors
            for (uint32_t n = 0; n < edge_per_node; n++) {
                const auto neighbor_index = neighbor_indices[n];
                const auto neighbor_weight = neighbor_weights[n];

                if(checkRNG(graph, edge_per_node, vertex_index, neighbor_index, neighbor_weight) == false) 
                    removed_rng_edges++;
            }
        }

        return removed_rng_edges;
    }

    /**
     * check if the graph is connected and contains only one graph component
     */
    static bool check_graph_connectivity(const deglib::search::SearchGraph& graph) {
        const auto node_count = graph.size();
        const auto edges_per_node = graph.getEdgesPerNode();

        // already checked nodes
        auto checked_ids = std::vector<bool>(node_count);

        // node the check
        auto check = std::vector<uint32_t>();

        // start with the first node
        checked_ids[0] = true;
        check.emplace_back(0);

        // repeat as long as we have nodes to check
		while(check.size() > 0) {	

            // neighbors which will be checked next round
            auto check_next = std::vector<uint32_t>();

            // get the neighbors to check next
            for (auto &&internal_index : check) {
                auto neighbor_indizes = graph.getNeighborIndices(internal_index);
                for (size_t e = 0; e < edges_per_node; e++) {
                    auto neighbor_index = neighbor_indizes[e];

                    if(checked_ids[neighbor_index] == false) {
                        checked_ids[neighbor_index] = true;
                        check_next.emplace_back(neighbor_index);
                    }
                }
            }

            check = std::move(check_next);
        }

        // how many nodes have been checked
        uint32_t checked_node_count = 0;
        for (size_t i = 0; i < node_count; i++)
            if(checked_ids[i])
                checked_node_count++;

        return checked_node_count == node_count;
    }
    
} // end namespace deglib::analysis