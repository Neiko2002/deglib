#pragma once

#include <cstdint>
#include <queue>
#include <unordered_set>

namespace deglib {

//#pragma pack(2)

  class ObjectDistance {          
    uint32_t            id_;
    float         distance_;
  
  public:

    ObjectDistance(const uint32_t id, const float distance): id_(id), distance_(distance) {}

    inline const uint32_t getId() const { return id_; } 

    inline const float getDistance() const { return distance_; } 

    inline bool operator==(const ObjectDistance &o) const {
      return (distance_ == o.distance_) && (id_ == o.id_);
    }

    inline bool operator<(const ObjectDistance &o) const {
      if (distance_ == o.distance_) {
        return id_ < o.id_;
      } else {
        return distance_ < o.distance_;
      }
    }

    inline bool operator>(const ObjectDistance &o) const {
      if (distance_ == o.distance_) {
        return id_ > o.id_;
      } else {
        return distance_ > o.distance_;
      }
    }
  };

//#pragma pack()

  // search result set containing node ids and distances
  typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance> > ResultSet;

  // set of unchecked node ids
  typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::greater<ObjectDistance> > UncheckedSet;

  /**
   * graph: search thru the edges and nodes of this graph
   * entry_nodes: graph nodes where to start the search and there distance to the query
   * query: a feature vector which might not be in the graph
   * l2space: distance calculation for two feature vectors
   * eps: parameter for the search
   * k: the amount of similar nodes which should be returned
   */ 
  deglib::ResultSet yahooSearch(const deglib::Graph &graph, const deglib::DynamicFeatureRepository &repository, const std::vector<deglib::ObjectDistance> &entry_nodes, const float* query, const deglib::L2Space &l2space, const float eps, const int k) {
    const auto dist_func = l2space.get_dist_func();
    const auto dist_func_param = l2space.get_dist_func_param();
    int dist_calcs = 0;

    // set of checked node ids
    auto checked_ids = tsl::robin_set<uint32_t>();
    checked_ids.reserve(k*50);  // estimated size 3000 to 5000 with k = 100
    for (auto &node : entry_nodes)
      checked_ids.insert(node.getId());

    // items to traverse next, start with the initial entry nodes
    auto next_nodes = deglib::UncheckedSet();
    for (auto &node : entry_nodes) 
      next_nodes.push(node);

    // result set
    auto results = deglib::ResultSet();
    for (auto &node : entry_nodes) 
      results.push(node);

    // search radius
    auto r = FLT_MAX;

    // iterate as long as good elements are in the next_nodes queue
    while(next_nodes.empty() == false) {

      // next node to check
      const auto next_node = next_nodes.top();
      next_nodes.pop();

      // max distance reached
      if(next_node.getDistance() > r * (1 + eps))
        break;

      // traverse never seen nodes
      const auto edges = graph.edges(next_node.getId());
      for (auto &edge : edges) {
        const auto neighbor_id = edge.first;

        // check if this node was not searched before
        if(checked_ids.insert(neighbor_id).second) {
          const auto neighbor_feature = repository.getFeature(neighbor_id);
          const auto neighbor_distance = dist_func(query, neighbor_feature, dist_func_param);
          //dist_calcs++;

          // check the neighborhood of this node later, if its good enough
          if(neighbor_distance <= r * (1 + eps)) {
            const auto candidate = deglib::ObjectDistance(neighbor_id, neighbor_distance);
            next_nodes.push(candidate);

            // remember the node, if its better than the worst in the result list
            if(neighbor_distance < r) {
              results.push(candidate);

              // update the search radius
              if(results.size() > k) {
                results.pop();
                r = results.top().getDistance();
              }							
            }
          }
        }
      }
    }

    //fmt::print("{} checked_ids, {} next_nodes, {} dist_calcs\n", checked_ids.size(), next_nodes.size(), dist_calcs);

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
  deglib::ResultSet yahooSearch(const deglib::Graph &graph, const deglib::DynamicFeatureRepository &repository, const std::vector<uint32_t> &entry_node_ids, const float* query, const deglib::L2Space &l2space, const float eps, const int k) {
    const auto dist_func = l2space.get_dist_func();
    const auto dist_func_param = l2space.get_dist_func_param();

    auto entry_nodes = std::vector<deglib::ObjectDistance>();
    entry_nodes.reserve(entry_node_ids.size());
    for (auto &&id : entry_node_ids) {
      const auto feature = repository.getFeature(id);
      const auto distance = dist_func(query, feature, dist_func_param);
      entry_nodes.push_back(deglib::ObjectDistance(id, distance));
    }
    
    return yahooSearch(graph, repository, entry_nodes, query, l2space, eps, k);
  }

}  // namespace deglib