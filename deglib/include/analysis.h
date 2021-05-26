#pragma once

#include <fmt/core.h>

#include "search.h"

namespace deglib::analysis
{
    /**
     * Check if the number of nodes and edges is consistent. 
     * The edges of a node should only contain unique neighbor indizies in ascending order and not a self-loop to the node.
     */
    static bool validation_check(const deglib::search::SearchGraph& graph, const uint32_t expected_nodes) {

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

                last_index = neighbor_index;
            }
        }
        
        return true;
    }

    
} // end namespace deglib::analysis