#pragma once

#include <fmt/core.h>
#include <tsl/robin_set.h>

#include "search.h"
#include "graph.h"

namespace deglib::analysis
{
    /**
     * Check if the number of nodes and edges is consistent. 
     * The edges of a node should only contain unique neighbor indizies in ascending order and not a self-loop to the node.
     * 
     * @param check_back_link checks if all the neighbors have the node in their neighbor-list (quite expensive)
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
            auto neighbor_indizies = graph.getNeighborIndizies(n);

            // check if the neighbor indizizes of the nodes are in ascending order and unique
            int64_t last_index = -1;
            for (size_t e = 0; e < edges_per_node; e++) {
                auto neighbor_index = neighbor_indizies[e];

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
    static float calc_graph_quality(const deglib::graph::MutableGraph& graph) {
        float total_distance = 0;
        uint32_t count = 0;

        const auto edges_per_node = graph.getEdgesPerNode();
        const auto node_count = graph.size();
        for (uint32_t n = 0; n < node_count; n++) {
            const auto weights = graph.getNeighborWeights(n);
            for (size_t e = 0; e < edges_per_node; e++)
                total_distance += weights[e];
            count += edges_per_node;
        }
        
        total_distance /= count;
        return total_distance;
    }

    /**
     * check if the graph is connected and contains only one graph component
     */
    static boolean check_graph_connectivity(const deglib::search::SearchGraph& graph) {
        const auto node_count = graph.size();
        const auto edges_per_node = graph.getEdgesPerNode();

        // already checked nodes
        auto checked_nodes = tsl::robin_set<uint32_t>();
        checked_nodes.reserve(node_count);

        // node the check
        auto check = std::vector<uint32_t>();

        // start with the first node
        checked_nodes.emplace(0);
        check.emplace_back(0);

        // repeat as long as we have nodes to check
		while(check.size() > 0) {	

            // neighbors which will be checked next round
            auto check_next = std::vector<uint32_t>();

            // get the neighbors to check next
            for (auto &&internal_index : check) {
                auto neighbor_indizes = graph.getNeighborIndizies(internal_index);
                for (size_t e = 0; e < edges_per_node; e++) {
                    auto neighbor_index = neighbor_indizes[e];

                    if(checked_nodes.emplace(neighbor_index).second)
                        check_next.emplace_back(neighbor_index);
                }
            }

            check = std::move(check_next);
        }

        return checked_nodes.size() == node_count;
    }
    
} // end namespace deglib::analysis