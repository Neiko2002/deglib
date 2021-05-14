#pragma once

#include <tsl/robin_set.h>

#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_set>

#include "distances.h"
#include "repository.h"
#include "graph.h"

namespace deglib
{


/**
 * graph: search thru the edges and nodes of this graph
 * entry_nodes: graph nodes where to start the search and there distance to the query
 * query: a feature vector which might not be in the graph
 * l2space: distance calculation for two feature vectors
 * eps: parameter for the search
 * k: the amount of similar nodes which should be returned
 */
deglib::ResultSet yahooSearch(const deglib::Graph& graph, const std::vector<deglib::ObjectDistance>& entry_nodes, 
                              const float* query, const deglib::L2Space& l2space, const float eps, const int k)
{
    const auto dist_func = l2space.get_dist_func();
    const auto dist_func_param = l2space.get_dist_func_param();

    // set of checked node ids
    auto checked_ids = std::vector<bool>(graph.size());
    for (auto& node : entry_nodes) checked_ids[node.getId()] = true;
    //auto checked_ids = tsl::robin_set<uint32_t>();
    //checked_ids.reserve(k * 50);  // estimated size 3000 to 5000 with k = 100
    //for (auto& node : entry_nodes) 
    //    checked_ids.insert(node.getId());

    // items to traverse next, start with the initial entry nodes
    auto internal_next_nodes = std::vector<ObjectDistance>();
    internal_next_nodes.reserve(1000);
    auto next_nodes = deglib::UncheckedSet(std::greater<ObjectDistance>(), std::move(internal_next_nodes));
    for (auto& node : entry_nodes) next_nodes.push(node);

    // result set
    auto internal_result = std::vector<ObjectDistance>();
    internal_result.reserve(k + 1);
    auto results = deglib::ResultSet(std::less<ObjectDistance>(), std::move(internal_result));
    for (auto& node : entry_nodes) results.push(node);

    // search radius
    auto r = std::numeric_limits<float>::max();

    // iterate as long as good elements are in the next_nodes queue
    auto good_neighbors = std::vector<deglib::Neighbor>(100); // this limits the max neighbor count to 100
    while (next_nodes.empty() == false)
    {
        // next node to check
        const auto next_node = next_nodes.top();
        next_nodes.pop();

        // max distance reached
        if (next_node.getDistance() > r * (1 + eps)) 
            break;

        good_neighbors.clear();
        for (auto& edge : graph.edges(next_node.getId())) {
            const auto &neighbor = edge.second;
            if (checked_ids[neighbor.id] == false)  {
                checked_ids[neighbor.id] = true;
                good_neighbors.emplace_back(std::move(neighbor));
            }
        }

        if (good_neighbors.empty())
            continue;

        _mm_prefetch((char *) good_neighbors[0].feature_vector, _MM_HINT_T0); 
         const auto good_neightbors_size = good_neighbors.size();
            for (size_t i = 0; i < good_neightbors_size; i++) {
            _mm_prefetch((char *) good_neighbors[std::min(i + 1, good_neightbors_size - 1)].feature_vector, _MM_HINT_T0); 

            const auto& neighbor = good_neighbors[i];       
            const auto neighbor_id = neighbor.id;
            const auto neighbor_distance = dist_func(query, neighbor.feature_vector, dist_func_param);

            // check the neighborhood of this node later, if its good enough
            if (neighbor_distance <= r * (1 + eps))
            {
                next_nodes.emplace(neighbor_id, neighbor_distance);

                // remember the node, if its better than the worst in the result list
                if (neighbor_distance < r)
                {
                    results.emplace(neighbor_id, neighbor_distance);

                    // update the search radius
                    if (results.size() > k)
                    {
                        results.pop();
                        r = results.top().getDistance();
                    }
                }
            }
        }
    }

    return results;
}

/**
 * graph: search thru the edges and nodes of this graph
 * entry_node_ids: id of graph nodes where to start the search
 * query: a feature vector which might not be in the graph
 * l2space: distance calculation for two feature vectors
 * eps: parameter for the search
 * k: the amount of similar nodes which should be returned
 */
deglib::ResultSet yahooSearch(const deglib::Graph& graph, const deglib::FeatureRepository& repository,
                              const std::vector<uint32_t>& entry_node_ids, const float* query,
                              const deglib::L2Space& l2space, const float eps, const int k)
{
    const auto dist_func = l2space.get_dist_func();
    const auto dist_func_param = l2space.get_dist_func_param();

    auto entry_nodes = std::vector<deglib::ObjectDistance>();
    entry_nodes.reserve(entry_node_ids.size());
    for (auto&& id : entry_node_ids)
    {
        const auto feature = repository.getFeature(id);
        const auto distance = dist_func(query, feature, dist_func_param);
        entry_nodes.push_back(deglib::ObjectDistance(id, distance));
    }

    return yahooSearch(graph, entry_nodes, query, l2space, eps, k);
}





// ----------------------------------------------------------------------------------------------
// -------------------------------- Static dataset and graph ------------------------------------
// ----------------------------------------------------------------------------------------------

/**
 * graph: search thru the edges and nodes of this graph
 * entry_nodes: graph nodes where to start the search and there distance to the query
 * query: a feature vector which might not be in the graph
 * l2space: distance calculation for two feature vectors
 * eps: parameter for the search
 * k: the amount of similar nodes which should be returned
 */
deglib::ResultSet yahooSearchStatic(const deglib::StaticGraph& graph, const std::vector<deglib::ObjectDistance>& entry_nodes, 
                                    const float* query, const deglib::L2Space& l2space, const float eps, const int k)
{
    const auto dist_func = l2space.get_dist_func();
    const auto dist_func_param = l2space.get_dist_func_param();

    // set of checked node ids
    auto checked_ids = std::vector<bool>(graph.size());
    for (auto& node : entry_nodes) checked_ids[node.getId()] = true;

    // items to traverse next, start with the initial entry nodes
    auto internal_next_nodes = std::vector<ObjectDistance>();
    internal_next_nodes.reserve(1000);
    auto next_nodes = deglib::UncheckedSet(std::greater<ObjectDistance>(), std::move(internal_next_nodes));
    for (auto& node : entry_nodes) next_nodes.push(node);

    // result set
    auto internal_result = std::vector<ObjectDistance>();
    internal_result.reserve(k + 1);
    auto results = deglib::ResultSet(std::less<ObjectDistance>(), std::move(internal_result));
    for (auto& node : entry_nodes) results.push(node);

    // search radius
    auto r = std::numeric_limits<float>::max();

    // iterate as long as good elements are in the next_nodes queue
    auto good_neighbors = std::vector<deglib::Neighbor>(100);       // this limits the max neighbor count to 100
    while (next_nodes.empty() == false)
    {
        // next node to check
        const auto next_node = next_nodes.top();
        next_nodes.pop();

        // max distance reached
        if (next_node.getDistance() > r * (1 + eps)) 
            break;

        good_neighbors.clear();
        for (auto& neighbor : graph.edges(next_node.getId())) {
            if (checked_ids[neighbor.id] == false) {
                checked_ids[neighbor.id] = true;
                good_neighbors.emplace_back(std::move(neighbor));
            }
        }

        if (good_neighbors.empty())
            continue;

        _mm_prefetch((char *) good_neighbors[0].feature_vector, _MM_HINT_T0); 
        const auto good_neightbors_size = good_neighbors.size();
        for (size_t i = 0; i < good_neightbors_size; i++) {
            _mm_prefetch((char *) good_neighbors[std::min(i + 1, good_neightbors_size - 1)].feature_vector, _MM_HINT_T0); 

            const auto& neighbor = good_neighbors[i];       
            const auto neighbor_id = neighbor.id;
            const auto neighbor_distance = dist_func(query, neighbor.feature_vector, dist_func_param);
            
            // check the neighborhood of this node later, if its good enough
            if (neighbor_distance <= r * (1 + eps))
            {
                next_nodes.emplace(neighbor_id, neighbor_distance);

                // remember the node, if its better than the worst in the result list
                if (neighbor_distance < r)
                {
                    results.emplace(neighbor_id, neighbor_distance);

                    // update the search radius
                    if (results.size() > k)
                    {
                        results.pop();
                        r = results.top().getDistance();
                    }
                }
            }
        }
    }

    return results;
}

/**
 * graph: search thru the edges and nodes of this graph
 * entry_node_ids: id of graph nodes where to start the search
 * query: a feature vector which might not be in the graph
 * l2space: distance calculation for two feature vectors
 * eps: parameter for the search
 * k: the amount of similar nodes which should be returned
 */
deglib::ResultSet yahooSearchStatic(const deglib::StaticGraph& graph, deglib::FeatureRepository& repository,
                                    const std::vector<uint32_t>& entry_node_ids, const float* query,
                                    const deglib::L2Space& l2space, const float eps, const int k)
{
    const auto dist_func = l2space.get_dist_func();
    const auto dist_func_param = l2space.get_dist_func_param();

    auto entry_nodes = std::vector<deglib::ObjectDistance>();
    entry_nodes.reserve(entry_node_ids.size());
    for (auto&& id : entry_node_ids)
    {
        const auto feature = repository.getFeature(id);
        const auto distance = dist_func(query, feature, dist_func_param);
        entry_nodes.push_back(deglib::ObjectDistance(id, distance));
    }

    return yahooSearchStatic(graph, entry_nodes, query, l2space, eps, k);
}

}  // namespace deglib