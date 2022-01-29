#pragma once

#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <functional>
#include <span>
#include <array>

#include <fmt/core.h>

#include "graph.h"

namespace deglib::builder
{
/*
struct C {
  std::deque<float> a;
  mutable std::mutex m;

  size_t size() const noexcept {
    std::scoped_lock l{m};
    return a.size();
  }
};
*/

struct BuilderAddTask {
  uint32_t label;
  uint64_t timestamp;
  std::vector<std::byte> feature;
};

struct BuilderRemoveTask {
  uint32_t label;
  uint64_t timestamp;
};

/**
 * Every graph change can be document with this struct. Needed to eventually revert back same changed.
 */ 
struct BuilderChange {
  uint32_t internal_index;
  uint32_t from_neighbor_index;
  float from_neighbor_weight;
  uint32_t to_neighbor_index;
  float to_neighbor_weight;
};

/**
 * Status of the build process. 
 * The process performs within a so called "step" a series of changes.
 * A step is either a series of graph improvement tries or the 
 * addition/deletion of a node followed be the improvement tries. 
 * The build process can only be stopped between two steps.
 */
struct BuilderStatus {
  uint64_t step;      // number of graph manipulation steps
  uint64_t added;     // number of added nodes
  uint64_t deleted;   // number of deleted nodes
  uint64_t improved;  // number of successful improvement
  uint64_t tries;     // number of improvement tries
};

class EvenRegularGraphBuilder {

    const uint8_t extend_k_;            // k value for extending the graph
    const float extend_eps_;            // eps value for extending the graph
    const bool extend_highLID_;
    const uint8_t improve_k_;           // k value for improving the graph
    const float improve_eps_;           // eps value for improving the graph
    const bool improve_highLID_;
    const uint8_t improve_step_factor_;
    const uint8_t max_path_length_;     // max amount of changes before canceling an improvement try
    const uint32_t swap_tries_;
    const uint32_t additional_swap_tries_;

    std::mt19937& rnd_;
    deglib::graph::MutableGraph& graph_;

    std::deque<BuilderAddTask> new_entry_queue_;
    std::queue<BuilderRemoveTask> remove_entry_queue_;

    // should the build loop run until the stop method is called
    bool stop_building_ = false;

  public:

    EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, std::mt19937& rnd, 
                            const uint8_t extend_k, const float extend_eps, const bool extend_highLID, 
                            const uint8_t improve_k, const float improve_eps, const bool improve_highLID, const uint8_t improve_step_factor = 2,
                            const uint8_t max_path_length = 10, const uint32_t swap_tries = 3, const uint32_t additional_swap_tries = 3) 
      : graph_(graph), rnd_(rnd), extend_k_(extend_k), extend_eps_(extend_eps), extend_highLID_(extend_highLID), 
        improve_k_(improve_k), improve_eps_(improve_eps), improve_highLID_(improve_highLID), improve_step_factor_(improve_step_factor),
        max_path_length_(max_path_length), swap_tries_(swap_tries), additional_swap_tries_(additional_swap_tries) {
    }

    EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, std::mt19937& rnd, const uint32_t swaps) 
      : EvenRegularGraphBuilder(graph, rnd, 
                                graph.getEdgesPerNode(), 0.2f, true,
                                graph.getEdgesPerNode(), 0.02f, false,
                                2, 10, swaps, swaps) {
    }

    EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, std::mt19937& rnd) 
      : EvenRegularGraphBuilder(graph, rnd, 1) {
    }

    /**
     * Provide the builder a new entry which it will append to the graph in the build() process.
     */ 
    void addEntry(const uint32_t label, std::vector<std::byte> feature) {
      auto time = std::chrono::system_clock::now();
      auto timestamp = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count());
      new_entry_queue_.emplace_back(label, timestamp, std::move(feature));
    }

    /**
     * Command the builder to remove a node from the graph as fast as possible.
     */ 
    void removeEntry(const uint32_t label) {
      auto time = std::chrono::system_clock::now();
      auto timestamp = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count());
      remove_entry_queue_.emplace(label, timestamp);
    }

  private:
  
    auto size() {
      return this->new_entry_queue_.size();
    }

    /**
     * Convert the queue into a vector with ascending distance order
     **/
    static auto topListAscending(deglib::search::ResultSet& queue) {
      const auto size = (int32_t) queue.size();
      auto topList = std::vector<deglib::search::ObjectDistance>(size);
      for (int32_t i = size - 1; i >= 0; i--) {
        topList[i] = std::move(const_cast<deglib::search::ObjectDistance&>(queue.top()));
        queue.pop();
      }
      return topList;
    }

    /**
     * Convert the queue into a vector with decending distance order
     **/
    static auto topListDescending(deglib::search::ResultSet& queue) {
      const auto size = queue.size();
      auto topList = std::vector<deglib::search::ObjectDistance>(size);
      for (size_t i = 0; i < size; i++) {
        topList[i] = std::move(const_cast<deglib::search::ObjectDistance&>(queue.top()));
        queue.pop();
      }
      return topList;
    }

    /**
     * The initial graph contains of "edges-per-node + 1" nodes.
     * Every node in this graph is connected to all other nodes.
     */
    void initialGraph(const std::span<const BuilderAddTask> entries) {
      auto& graph = this->graph_;
      const auto& feature_space = graph.getFeatureSpace();
      const auto& dist_func = feature_space.get_dist_func();
      const auto& dist_func_param = feature_space.get_dist_func_param();

      // compute a full distance matrix
      const auto size = entries.size();
      auto matrix = std::vector<std::vector<float>>(size);
      for (size_t y = 0; y < size; y++) {
        auto& row = matrix[y];
        const auto query = entries[y].feature.data();
        for (size_t x = 0; x < size; x++) {
          row.emplace_back(dist_func(query, entries[x].feature.data(), dist_func_param));
        }
      }

      // setup the nodes first, to be able to get their internal indizies
      for (auto &&entry : entries)   
        graph.addNode(entry.label, entry.feature.data());
      
      // setup the edges for every node
      const auto edges_per_node = graph.getEdgesPerNode();
      auto neighbors = std::vector<std::pair<uint32_t,float>>();
      auto neighbor_indizies = std::vector<uint32_t>(edges_per_node);
      auto neighbor_weights = std::vector<float>(edges_per_node);
      for (size_t entry_idx = 0; entry_idx < size; entry_idx++) {

        // gather the edge distances to the other nodes
        neighbors.clear();
        for (size_t i = 0; i < size; i++) {

          // skip the node to which we collect the edges for
          if(i == entry_idx) continue;

          const auto neighbor_index = graph.getInternalIndex(entries[i].label);
          const auto distance = matrix[entry_idx][i];
          neighbors.emplace_back(neighbor_index, distance);
        }
        
        // sort the edges by their internal index values
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
        neighbor_indizies.clear();
        neighbor_weights.clear();
        for (auto &&neighbor : neighbors) {
          neighbor_indizies.emplace_back(neighbor.first);
          neighbor_weights.emplace_back(neighbor.second);
        }
        
        // store the edges of the new node
        const auto internal_index = graph.getInternalIndex(entries[entry_idx].label);
        graph.changeEdges(internal_index, neighbor_indizies.data(), neighbor_weights.data());
      }
    }

    /**
     * Extend the graph with a new node. Find good existing node to which this new node gets connected.
     */
    void extendGraph(const BuilderAddTask& add_task) {
      auto& graph = this->graph_;
      const auto external_label = add_task.label;

      // graph should not contain a node with the same label
      if(graph.hasNode(external_label)) {
        fmt::print(stderr, "graph contains node {} already. can not add it again \n", external_label);
        perror("");
        abort();
      }

      // find good neighbors for the new node
      const auto new_node_feature = add_task.feature.data();
      const auto edges_per_node = graph.getEdgesPerNode();
      auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
      const std::vector<uint32_t> entry_node_indizies = { distrib(this->rnd_) };
      auto top_list = graph.search(entry_node_indizies, new_node_feature, this->extend_eps_, std::max(this->extend_k_, edges_per_node));
      const auto results = topListAscending(top_list);

      // their should always be enough neighbors (search results), otherwise the graph would be broken
      if(results.size() < edges_per_node) {
        fmt::print(stderr, "the graph search for the new node {} did only provided {} results \n", external_label, results.size());
        perror("");
        abort();
      }

      // add an empty node to the graph (no neighbor information yet)
      const auto internal_index = graph.addNode(external_label, new_node_feature);

      // for computing distances to neighbors not in the result queue
      const auto dist_func = graph.getFeatureSpace().get_dist_func();
      const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
     
      // remove an edge of the good neighbors and connect them with this new node
      auto new_neighbors = std::vector<std::pair<uint32_t, float>>();
      for (size_t i = 0; i < results.size() && new_neighbors.size() < edges_per_node; i++) {
        const auto& result = results[i];

        // check if the node is already in the edge list of the new node (added during a previous loop-run)
        // since all edges are undirected and the edge information of the new node does not yet exist, we search the other way around.
        if(graph.hasEdge(result.getInternalIndex(), internal_index)) 
          continue;

        // This version is good for high LID datasets or small graphs with low distance count limit during ANNS
        uint32_t new_neighbor_index = 0;
        float new_neighbor_distance = -1;
        if(extend_highLID_) {

          // find the worst edge of the new neighbor
          float new_neighbor_weight = -1;
          const auto neighbor_weights = graph.getNeighborWeights(result.getInternalIndex());
          const auto neighbor_indizies = graph.getNeighborIndizies(result.getInternalIndex());
          for (size_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {
            const auto neighbor_index = neighbor_indizies[edge_idx];
            const auto neighbor_weight = neighbor_weights[edge_idx];

            // the suggested neighbor might already be in the edge list of the new node
            // the weight of the neighbor might not be worst than the current worst one
            if(neighbor_weight > new_neighbor_weight && graph.hasEdge(neighbor_index, internal_index) == false) {
              new_neighbor_weight = neighbor_weight;
              new_neighbor_index = neighbor_index;
            }          
          }

          // new_neighbor_weight == -1 should not be possible, otherwise the new node is connected to every node in the neighbor-list of the result-node and still has space for more
          if(new_neighbor_weight != -1) 
            new_neighbor_distance = dist_func(add_task.feature.data(), graph.getFeatureVector(new_neighbor_index), dist_func_param); 
        }
        else
        {
          // find the edge which improves the distortion the most: (distance_new_edge1 + distance_new_edge2) - distance_removed_edge
          {
            float best_distortion = std::numeric_limits<float>::max();
            const auto neighbor_indizies = graph.getNeighborIndizies(result.getInternalIndex());
            const auto neighbor_weights = graph.getNeighborWeights(result.getInternalIndex());
            for (size_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {
              const auto neighbor_index = neighbor_indizies[edge_idx];
              if(graph.hasEdge(neighbor_index, internal_index) == false) {
                const auto neighbor_distance = dist_func(new_node_feature, graph.getFeatureVector(neighbor_index), dist_func_param);

                // take the neighbor with the best distance to the new node, which might already be in its edge list
                float distortion = (result.getDistance() + neighbor_distance) - neighbor_weights[edge_idx];
                if(distortion < best_distortion) {
                  best_distortion = distortion;
                  new_neighbor_index = neighbor_index;
                  new_neighbor_distance = neighbor_distance;
                }          
              }
            }
          }
        }

        // this should not be possible, otherwise the new node is connected to every node in the neighbor-list of the result-node and still has space for more
        if(new_neighbor_distance == -1) {
          fmt::print(stderr, "it was not possible to find a bad edge in the neighbor list of node {} which would connect to node {} \n", result.getInternalIndex(), internal_index);
          perror("");
          abort();
        }

        // place the new node in the edge list of the result-node
        graph.changeEdge(result.getInternalIndex(), new_neighbor_index, internal_index, result.getDistance());
        new_neighbors.emplace_back(result.getInternalIndex(), result.getDistance());

        // place the new node in the edge list of the best edge neighbor
        graph.changeEdge(new_neighbor_index, result.getInternalIndex(), internal_index, new_neighbor_distance);
        new_neighbors.emplace_back(new_neighbor_index, new_neighbor_distance);
      }

      // sort the neighbors by their neighbor indizies and store them in the new node
      {
        std::sort(new_neighbors.begin(), new_neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
        auto neighbor_indizies = std::vector<uint32_t>(new_neighbors.size());
        auto neighbor_weights = std::vector<float>(new_neighbors.size());
        for (size_t i = 0; i < new_neighbors.size(); i++) {
          const auto& neighbor = new_neighbors[i];
          neighbor_indizies[i] = neighbor.first;
          neighbor_weights[i] = neighbor.second;
        }
        graph.changeEdges(internal_index, neighbor_indizies.data(), neighbor_weights.data());  
      }

      // try to improve some of the non-perfect edges
      {
        auto nonperfect_neighbors = std::vector<std::pair<uint32_t, float>>();
        for (size_t i = 0; i < new_neighbors.size(); i++) {
          const auto& neighbor = new_neighbors[i];

          bool perfect = false;
          for (size_t r = 0; r < results.size(); r++) {
            const auto& result = results[r];
            if(result.getInternalIndex() == neighbor.first) {
              perfect = true;
              break;
            }
          }

          if(perfect == false && graph.hasEdge(internal_index, neighbor.first)) 
            nonperfect_neighbors.emplace_back(neighbor.first, neighbor.second);
        }

        std::sort(nonperfect_neighbors.begin(), nonperfect_neighbors.end(), [](const auto& x, const auto& y){return x.second < y.second;});
        // for (size_t i = 0; i < nonperfect_neighbors.size(); i++) // all non perfect edges
        for (size_t i = 0; i < nonperfect_neighbors.size() / 2; i++) // first half of non perfect edges
          if(graph.hasEdge(internal_index, nonperfect_neighbors[i].first)) 
            improveEdges(internal_index, nonperfect_neighbors[i].first, nonperfect_neighbors[i].second);
      }
    }

    /**
     * Removing a node from the graph.
     */
    void shrinkGraph(const BuilderRemoveTask& del_task) {
      fmt::print(stderr, "shrinking the graph by node {} is not yet implemented \n", del_task.label);
      perror("");
      abort();
    }

    /**
     * Do not call this method directly instead call improve() to improve the graph.
     *  
     * This is the extended part of the optimization process.
     * The method takes an array where all graph changes will be documented.
	   * Node1 and node2 might be in a separate subgraph than node3 and node4.
     * Thru a series of edges swaps both subgraphs should be reconnected..
     * If those changes improve the graph this method returns true otherwise false. 
     * 
     * @return true if a good sequences of changes has been found
     */
    bool improveEdges(std::vector<deglib::builder::BuilderChange>& changes, uint32_t node1, uint32_t node2, uint32_t node3, uint32_t node4, float total_gain, const uint8_t steps) {
      auto& graph = this->graph_;
      const auto edges_per_node = graph.getEdgesPerNode();
      
      // the settings are the same for the first two iterations
      const auto high_variance_swaps = this->improve_highLID_;
      const auto search_eps = this->improve_eps_; 
      const auto search_k = this->improve_k_ - (uint8_t) std::max(0, (steps-1)*this->improve_step_factor_);

      if(high_variance_swaps) {
    
        // 1. Find a edge for node2 which connects to the subgraph of node3 and node4. 
        //    Consider only nodes of the approximate nearest neighbor search. Since the 
        //    search started from node3 and node4 all nodes in the result list are in 
        //    their subgraph and would therefore connect the two potential subgraphs.	
        float dist23 = 0;
        {
          const auto node2_feature = graph.getFeatureVector(node2);
          const std::vector<uint32_t> entry_node_indizies = { node3, node4 };
          auto top_list = graph.search(entry_node_indizies, node2_feature, search_eps, search_k);

          // find a good new node3
          for(auto&& result : topListAscending(top_list)) {

            // TODO maybe making sure the new node3 is not the old node3 or even node4 helps
            if(node1 != result.getInternalIndex() && node2 != result.getInternalIndex() && graph.hasEdge(node2, result.getInternalIndex()) == false) {
              node3 = result.getInternalIndex();
              dist23 = result.getDistance();
              break;
            }
          }

          // no new node3 was found
          if(dist23 == 0)
            return false;

          // replace the temporary self-loop of node2 with a connection to node3. 
          graph.changeEdge(node2, node2, node3, dist23);
          changes.emplace_back(node2, node2, 0.f, node3, dist23);
          total_gain -= dist23;
        }

        // 2. All nodes are connected but the subgraph between node1/node2 and node3/node4 might just have one edge(node2, node3).
        //    Furthermore Node 3 has now to many edges, remove the worst one. Ignore the just added edge. 
        //    FYI: If the just selected node3 is the same as the old node3, this process might cut its connection to node4 again.
        //    This will be fixed in the next step or until the recursion reaches max_path_length.
        float dist34 = 0;
        {
          // 2.1 find the worst edge of node3
          uint32_t bad_neighbor_index = 0;
          float bad_neighbor_weight = 0.f;
          const auto neighbor_weights = graph.getNeighborWeights(node3);
          const auto neighbor_indizies = graph.getNeighborIndizies(node3);
          for (size_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {

            // do not remove the edge which was just added
            if(neighbor_indizies[edge_idx] != node2 && bad_neighbor_weight < neighbor_weights[edge_idx]) {
              bad_neighbor_index = neighbor_indizies[edge_idx];
              bad_neighbor_weight = neighbor_weights[edge_idx];    
            }
          }

          // 2.2 Remove the worst edge of node3 to node4 and replace it with the connection to node2
          //     Add a temporary self-loop for node4 for the missing edge to node3
          node4 = bad_neighbor_index;
          dist34 = bad_neighbor_weight;
          total_gain += dist34;
          graph.changeEdge(node3, node4, node2, dist23);
          changes.emplace_back(node3, node4, dist34, node2, dist23);
          graph.changeEdge(node4, node3, node4, 0.f);
          changes.emplace_back(node4, node3, dist34, node4, 0.f);
        }

      }
      else
      {
      
        // 1. Find an edge for node2 which connects to the subgraph of node3 and node4. 
        //    Consider only nodes of the approximate nearest neighbor search. Since the 
        //    search started from node3 and node4 all nodes in the result list are in 
        //    their subgraph and would therefore connect the two potential subgraphs.	
        {
          const auto node2_feature = graph.getFeatureVector(node2);
          const std::vector<uint32_t> entry_node_indizies = { node3, node4 };
          auto top_list = graph.search(entry_node_indizies, node2_feature, search_eps, search_k);

          // find a good new node3
          float best_gain = total_gain;
          float dist23 = -1;
          float dist34 = -1;

          // We use the descending order to find the worst swap combination with the best gain
          // Sometimes the gain between the two best combinations is the same, its better to use one with the bad edges to make later improvements easier
          for(auto&& result : topListDescending(top_list)) {

            // node1 and node2 got tested in the recursive call before and node4 got just disconnected from node2
            if(node1 != result.getInternalIndex() && node2 != result.getInternalIndex() && graph.hasEdge(node2, result.getInternalIndex()) == false) {
              uint32_t new_node3 = result.getInternalIndex();

              // 1.1 When node2 and the new node 3 gets connected full graph connectivity is assured again, 
              //     but the subgraph between node1/node2 and node3/node4 might just have one edge(node2, node3).
              //     Furthermore Node 3 has now to many edges, find an good edge to remove to improve the overall graph distortion. 
              //     FYI: If the just selected node3 is the same as the old node3, this process might cut its connection to node4 again.
              //     This will be fixed in the next step or until the recursion reaches max_path_length.
              const auto neighbor_weights = graph.getNeighborWeights(new_node3);
              const auto neighbor_indizies = graph.getNeighborIndizies(new_node3);
              for (size_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {
                uint32_t new_node4 = neighbor_indizies[edge_idx];

                // compute the gain of the graph distortion if this change would be applied
                const auto gain = total_gain - result.getDistance() + neighbor_weights[edge_idx];

                // do not remove the edge which was just added
                if(new_node4 != node2 && best_gain < gain) {
                  best_gain = gain;
                  node3 = new_node3;
                  node4 = new_node4;
                  dist23 = result.getDistance();
                  dist34 = neighbor_weights[edge_idx];    
                }
              }
            }
          }

          // no new node3 was found
          if(dist23 == -1)
            return false;

          // replace the temporary self-loop of node2 with a connection to node3. 
          total_gain = (total_gain - dist23) + dist34;
          graph.changeEdge(node2, node2, node3, dist23);
          changes.emplace_back(node2, node2, 0.f, node3, dist23);

          // 1.2 Remove the worst edge of node3 to node4 and replace it with the connection to node2
          //     Add a temporaty self-loop for node4 for the missing edge to node3
          graph.changeEdge(node3, node4, node2, dist23);
          changes.emplace_back(node3, node4, dist34, node2, dist23);
          graph.changeEdge(node4, node3, node4, 0.f);
          changes.emplace_back(node4, node3, dist34, node4, 0.f);
        }

        // There is no step 2, since step 1 and 2 got combined in the method above with a different heuristic
      }

      // 3. Try to connect node1 with node4
      {
        const auto& feature_space = this->graph_.getFeatureSpace();
        const auto dist_func = feature_space.get_dist_func();
        const auto dist_func_param = feature_space.get_dist_func_param();

        // 3.1a Node1 and node4 might be the same. This is quite the rare case, but would mean there are two edges missing.
        //     Proceed like extending the graph:
        //     Search for a good node to connect to, remove its worst edge and connect
        //     both nodes of the worst edge to the node4. Skip the edge any of the two
        //     two nodes are already connected to node4.
        if(node1 == node4) {

          // finds and keeps the best possible connection for node 4, 
          // even if other nodes do not get ideal connections with this trade
          if(high_variance_swaps) {
  
            // find a good (not yet connected) node for node1/node4
            const std::vector<uint32_t> entry_node_indizies = { node2, node3 };
            const auto node4_feature = graph.getFeatureVector(node4);
            auto top_list = graph.search(entry_node_indizies, node4_feature, search_eps, search_k);

            for(auto&& result : topListAscending(top_list)) {
              const auto good_node = result.getInternalIndex();

              // the new node should not be connected to node4 yet
              if(node4 != good_node && graph.hasEdge(node4, good_node) == false) {
                const auto good_node_dist = result.getDistance();

                // select any edge of the good node which improves the graph quality when replaced with a connection to node 4
                const auto neighbors_indizies = graph.getNeighborIndizies(good_node);
                const auto neighbor_weights = graph.getNeighborWeights(good_node);
                for (size_t i = 0; i < edges_per_node; i++) {
                  const auto selected_neighbor = neighbors_indizies[i];

                  // ignore edges where the second node is already connect to node4
                  if(node4 != selected_neighbor && graph.hasEdge(node4, selected_neighbor) == false) {
                    const auto old_neighbor_dist = neighbor_weights[i];
                    const auto new_neighbor_dist = dist_func(node4_feature, graph.getFeatureVector(selected_neighbor), dist_func_param);

                    // do all the changes improve the graph?
                    if((total_gain + old_neighbor_dist) - (good_node_dist + new_neighbor_dist) > 0) {

                      // replace the two self-loops of node4/node1 with a connection to the good node and its selected neighbor
                      graph.changeEdge(node4, node4, good_node, good_node_dist);
                      changes.emplace_back(node4, node4, 0.f, good_node, good_node_dist);
                      graph.changeEdge(node4, node4, selected_neighbor, new_neighbor_dist);
                      changes.emplace_back(node4, node4, 0.f, selected_neighbor, new_neighbor_dist);

                      // replace from good node the connection to the selected neighbor with one to node4
                      graph.changeEdge(good_node, selected_neighbor, node4, good_node_dist);
                      changes.emplace_back(good_node, selected_neighbor, old_neighbor_dist, node4, good_node_dist);

                      // replace from the selected neighbor the connection to the good node with one to node4
                      graph.changeEdge(selected_neighbor, good_node, node4, new_neighbor_dist);
                      changes.emplace_back(selected_neighbor, good_node, old_neighbor_dist, node4, new_neighbor_dist);

                      return true;
                    }
                  }
                }
              }
            }
          } 
          else
          {
            // find a good (not yet connected) node for node1/node4
            const std::vector<uint32_t> entry_node_indizies = { node2, node3 };
            const auto node4_feature = graph.getFeatureVector(node4);
            auto top_list = graph.search(entry_node_indizies, node4_feature, search_eps, search_k);

            float best_gain = 0;
            uint32_t best_selected_neighbor = 0;
            float best_old_neighbor_dist = 0;
            float best_new_neighbor_dist = 0;
            uint32_t best_good_node = 0;
            float best_good_node_dist = 0;
            for(auto&& result : topListAscending(top_list)) {
              const auto good_node = result.getInternalIndex();

              // the new node should not be connected to node4 yet
              if(node4 != good_node && graph.hasEdge(node4, good_node) == false) {
                const auto good_node_dist = result.getDistance();

                // select any edge of the good node which improves the graph quality when replaced with a connection to node 4
                const auto neighbors_indizies = graph.getNeighborIndizies(good_node);
                const auto neighbor_weights = graph.getNeighborWeights(good_node);
                for (size_t i = 0; i < edges_per_node; i++) {
                  const auto selected_neighbor = neighbors_indizies[i];

                  // ignore edges where the second node is already connect to node4
                  if(node4 != selected_neighbor && graph.hasEdge(node4, selected_neighbor) == false) {
                    const auto old_neighbor_dist = neighbor_weights[i];
                    const auto new_neighbor_dist = dist_func(node4_feature, graph.getFeatureVector(selected_neighbor), dist_func_param);

                    // do all the changes improve the graph?
                    float new_gain = (total_gain + old_neighbor_dist) - (good_node_dist + new_neighbor_dist);
                    if(best_gain < new_gain) {
                      best_gain = new_gain;
                      best_selected_neighbor = selected_neighbor;
                      best_old_neighbor_dist = old_neighbor_dist;
                      best_new_neighbor_dist = new_neighbor_dist;
                      best_good_node = good_node;
                      best_good_node_dist = good_node_dist;
                    }
                  }
                }
              }
            }

            if(best_gain > 0)
            {

              // replace the two self-loops of node4/node1 with a connection to the good node and its selected neighbor
              graph.changeEdge(node4, node4, best_good_node, best_good_node_dist);
              changes.emplace_back(node4, node4, 0.f, best_good_node, best_good_node_dist);
              graph.changeEdge(node4, node4, best_selected_neighbor, best_new_neighbor_dist);
              changes.emplace_back(node4, node4, 0.f, best_selected_neighbor, best_new_neighbor_dist);

              // replace from good node the connection to the selected neighbor with one to node4
              graph.changeEdge(best_good_node, best_selected_neighbor, node4, best_good_node_dist);
              changes.emplace_back(best_good_node, best_selected_neighbor, best_old_neighbor_dist, node4, best_good_node_dist);

              // replace from the selected neighbor the connection to the good node with one to node4
              graph.changeEdge(best_selected_neighbor, best_good_node, node4, best_new_neighbor_dist);
              changes.emplace_back(best_selected_neighbor, best_good_node, best_old_neighbor_dist, node4, best_new_neighbor_dist);

              return true;
            }
          }


        } else {

          // 3.1b If there is a way from node2 or node3, to node1 or node4 then ...
				  //      Try to connect node1 with node4
          //      Much more likly than 3.1a 
				  if(graph.hasEdge(node1, node4) == false) {

            // Is the total of all changes still beneficial?
            const auto dist14 = dist_func(graph.getFeatureVector(node1), graph.getFeatureVector(node4), dist_func_param);
            if((total_gain - dist14) > 0) {

              const std::vector<uint32_t> entry_node_indizies = { node2, node3 }; 
              if(graph.hasPath(entry_node_indizies, node1, this->improve_eps_, this->improve_k_).size() > 0 || graph.hasPath(entry_node_indizies, node4, this->improve_eps_, improve_k_).size() > 0) {
                
                // replace the the self-loops of node1 with a connection to the node4
                graph.changeEdge(node1, node1, node4, dist14);
                changes.emplace_back(node1, node1, 0.f, node4, dist14);

                // replace the the self-loops of node4 with a connection to the node1
                graph.changeEdge(node4, node4, node1, dist14);
                changes.emplace_back(node4, node4, 0.f, node1, dist14);

                return true;
              }
            }
          }
        }
      }


      
      // 4. Maximum path length
      if(steps >= this->max_path_length_ || (this->improve_k_ - (steps+1)*improve_step_factor_) <= 1) {
        //fmt::print("Reached maxiumum path length without improvements. Rollback.\n");	
        return false;
      }
      
      // 5. swap node1 and node4 every second round, to give each a fair chance
      if(steps % 2 == 1) {
        uint32_t b = node1;
        node1 = node4;
        node4 = b;
      }

      // 6. early stop
      // TODO Since an edge is missing the total gain should always be higher zero. A better heuristic would be total_gain-<current worst edge of node4>.
      // In the next iteration we try to find a good edge for node 4 which is probably better than the current worst edge of node 4.
      // But since the total gain could already be so bad that even if the new edge is better, the overall gain would still be negative, we should stop here.
      if(total_gain < 0) {
        //fmt::print("Current swap path is degenerating the graph. Rollback.\n");
        return false;
      }

      return improveEdges(changes, node1, node4, node2, node3, total_gain, steps + 1);
    }

    bool improveEdges() {

      auto& graph = this->graph_;
      const auto edges_per_node = graph.getEdgesPerNode();

      // 1. remove the worst edge of a random node 

      // 1.1 select a random node
      auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
      uint32_t node1 = distrib(this->rnd_);

      // 1.2 find the worst edge of this node
      uint32_t bad_neighbor_index = 0;
      float bad_neighbor_weight = 0.f;
      const auto neighbor_weights = graph.getNeighborWeights(node1);
      const auto neighbor_indizies = graph.getNeighborIndizies(node1);
      for (size_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {
        if(bad_neighbor_weight < neighbor_weights[edge_idx]) {
          bad_neighbor_index = neighbor_indizies[edge_idx];
          bad_neighbor_weight = neighbor_weights[edge_idx];    
        }
      }

      return improveEdges(node1, bad_neighbor_index, bad_neighbor_weight);
    }

    bool improveEdges(uint32_t node1, uint32_t node2, float dist12) {
      auto changes = std::vector<deglib::builder::BuilderChange>();

      // remove the edge between node 1 and node 2 (add temporary self-loops)
      auto& graph = this->graph_;
      graph.changeEdge(node1, node2, node1, 0.f);
      changes.emplace_back(node1, node2, dist12, node1, 0.f);
      graph.changeEdge(node2, node1, node2, 0.f);
      changes.emplace_back(node2, node1, dist12, node2, 0.f);

      if(improveEdges(changes, node1, node2, node1, node1, dist12, 0) == false) {

        // undo all changes, in reverse order
        const auto size = changes.size();
        for (size_t i = 0; i < size; i++) {
          auto c = changes[(size - 1) - i];
          this->graph_.changeEdge(c.internal_index, c.to_neighbor_index, c.from_neighbor_index, c.from_neighbor_weight);
        }

        return false;
      }

      return true;
    }

  public:

    /**
     * Build the graph. This could be run on a separate thread in an infinite loop.
     */
    auto& build(std::function<void(deglib::builder::BuilderStatus&)> callback, const bool infinite = false) {
      auto status = BuilderStatus{};
      const auto edge_per_node = this->graph_.getEdgesPerNode();

      // try to build an initial graph, containing the minium amount of nodes (edge_per_node + 1)
      const auto edge_per_node_p1 = (uint8_t)(edge_per_node + 1);
      if(graph_.size() < edge_per_node_p1) {

        // graph should be empty to initialize
        if(this->graph_.size() > 0) {
          fmt::print(stderr, "graph has already {} nodes and can therefore not be initialized \n", this->graph_.size());
          perror("");
          abort();
        }

        // wait until enough new entries exists to build the initial graph
        while(new_entry_queue_.size() < edge_per_node_p1)
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        // setup the initial graph
        {
          std::array<BuilderAddTask, std::numeric_limits<uint8_t>::max()> initial_entries;
          std::copy(new_entry_queue_.begin(), std::next(new_entry_queue_.begin(), edge_per_node_p1), initial_entries.begin());
          new_entry_queue_.erase(new_entry_queue_.begin(), std::next(new_entry_queue_.begin(), edge_per_node_p1));
          initialGraph({initial_entries.data(), edge_per_node_p1});
        }

        // inform the callback about the initial graph
        status.added += edge_per_node_p1;
        callback(status);
      } 
      else 
      {
        status.added = graph_.size();
      }

      // run a loop to add, delete and improve the graph
      do{

        // add or delete a node
        if(this->new_entry_queue_.size() > 0 || this->remove_entry_queue_.size() > 0) {
          auto add_task_timestamp = std::numeric_limits<uint64_t>::max();
          auto del_task_timestamp = std::numeric_limits<uint64_t>::max();

          if(this->new_entry_queue_.size() > 0) 
            add_task_timestamp = this->new_entry_queue_.front().timestamp;

          if(this->remove_entry_queue_.size() > 0) 
            del_task_timestamp = this->remove_entry_queue_.front().timestamp;

          if(add_task_timestamp < del_task_timestamp) {
            extendGraph(this->new_entry_queue_.front());
            status.added++;
            this->new_entry_queue_.pop_front();
          } else {
            shrinkGraph(this->remove_entry_queue_.front());
            status.deleted++;
            this->remove_entry_queue_.pop();
          }
        }

        //try to improve the graph
        if(improve_k_ > 0) {
          for (int64_t swap_try = 0; swap_try < int64_t(this->swap_tries_); swap_try++) {
            status.tries++;

            if(this->improveEdges()) {
              status.improved++;
              swap_try -= this->additional_swap_tries_;
            }
          }
        }
        
        status.step++;
        callback(status);
      }
      while(this->stop_building_ == false && (infinite || this->new_entry_queue_.size() > 0 || this->remove_entry_queue_.size() > 0));

      // return the finished graph
      return this->graph_;
    }

    /**
     * Stop the build process
     */
    void stop() {
      this->stop_building_ = true;
    }
};

} // end namespace deglib::builder