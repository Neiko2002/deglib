#pragma once

#include <random>
#include <chrono>
#include <thread>
#include <algorithm>

#include <fmt/core.h>

#include "graph.h"

namespace deglib::builder
{

struct BuilderAddTask {
  uint32_t label;
  std::vector<std::byte> feature;
  uint64_t timestamp = std::numeric_limits<uint64_t>::max();
};

struct BuilderRemoveTask {
  uint32_t label;
  uint64_t timestamp = std::numeric_limits<uint64_t>::max();
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

    const uint8_t extend_k_;         // k value for extending the graph
    const float extend_eps_;          // eps value for extending the graph
    const uint8_t improve_k_;        // k value for improving the graph
    const float improve_eps_;         // eps value for improving the graph
    const uint8_t max_path_length_;  // max amount of changes before canceling an improvement try
    const uint32_t swap_tries_;
    const uint32_t additional_swap_tries_;

    std::mt19937& rnd_;
    deglib::graph::MutableGraph& graph_;

    std::queue<BuilderAddTask> new_entry_queue_;
    std::queue<BuilderRemoveTask> remove_entry_queue_;

    // should the build loop run until the stop method is called
    bool stop_building_ = false;

  public:

    EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, const uint8_t extend_k, const float extend_eps, const uint8_t improve_k, const float improve_eps, const uint8_t max_path_length, const uint32_t swap_tries, const uint32_t additional_swap_tries, std::mt19937& rnd) 
      : graph_(graph), extend_k_(extend_k), extend_eps_(extend_eps), improve_k_(improve_k), improve_eps_(improve_eps), max_path_length_(max_path_length), swap_tries_(swap_tries), additional_swap_tries_(additional_swap_tries), rnd_(rnd) {
    }

    /**
     * Provide the builder a new entry which it will append to the graph in the build() process.
     */ 
    void addEntry(uint32_t label, std::vector<std::byte> feature) {
      auto time = std::chrono::system_clock::now();
      auto timestamp = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count());
      new_entry_queue_.emplace(label, std::move(feature), timestamp);
    }

    /**
     * Command the builder to remove a node from the graph as fast as possible.
     */ 
    void removeEntry(uint32_t label) {
      auto time = std::chrono::system_clock::now();
      auto timestamp = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(time.time_since_epoch()).count());
      remove_entry_queue_.emplace(label, timestamp);
    }


  private:
  
    auto size() {
      return this->new_entry_queue_.size();
    }

    /**
     * The initial graph contains of "edges-per-node + 1" nodes.
     * Every node in this graph is connected to all other nodes.
     */
    void initialGraph(const std::vector<BuilderAddTask> entries) {
      const auto& feature_space = this->graph_.getFeatureSpace();
      const auto dist_func = feature_space.get_dist_func();
      const auto dist_func_param = feature_space.get_dist_func_param();

      // compute a full distance matrix
      auto size = entries.size();
      auto matrix = std::vector<std::vector<float>>(size);
      for (size_t y = 0; y < size; y++) {
        auto& row = matrix[y];
        auto query = entries[y].feature.data();
        for (size_t x = 0; x < size; x++) {
          row.emplace_back(dist_func(query, entries[x].feature.data(), dist_func_param));
        }
      }

      // setup the nodes first, to be able to get their internal indizies
      auto edges_per_node = this->graph_.getEdgesPerNode();
      for (auto &&entry : entries)   
        this->graph_.addNode(entry.label, entry.feature.data());
      
      // setup the edges for every node
      auto neighbors = std::vector<std::pair<uint32_t,float>>();
      auto neighbor_indizies = std::vector<uint32_t>(edges_per_node);
      auto neighbor_weights = std::vector<float>(edges_per_node);
      for (size_t entry_idx = 0; entry_idx < size; entry_idx++) {

        // gather the edge distances to the other nodes
        neighbors.clear();
        for (size_t i = 0; i < size; i++) {

          // skip the node to which we collect the edges for
          if(i == entry_idx) continue;

          auto &&neighbor = entries[i];
          auto neighbor_index = this->graph_.getInternalIndex(neighbor.label);
          auto distance = matrix[entry_idx][i];
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
        auto &&entry = entries[entry_idx];
        auto internal_index = this->graph_.getInternalIndex(entry.label);
        this->graph_.changeEdges(internal_index, neighbor_indizies.data(), neighbor_weights.data());
      }
    }

    /**
     * Extend the graph with a new node. Find good existing node to which this new node gets connected.
     */
    void extendGraph(BuilderAddTask add_task) {
      const auto external_label = add_task.label;

      // graph should not contain a node with the same label
      if(this->graph_.hasNode(external_label)) {
        fmt::print(stderr, "graph contains node {} already. can not add it again \n", external_label);
        perror("");
        abort();
      }

      // find good neighbors for the new node
      const auto edges_per_node = this->graph_.getEdgesPerNode();
      const auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(this->graph_.size() - 1));
      const std::vector<uint32_t> entry_node_indizies = { distrib(this->rnd_) };
      auto result_queue = this->graph_.yahooSearch(entry_node_indizies, add_task.feature.data(), this->extend_eps_, std::max(this->extend_k_, edges_per_node));

      // their should always be enough neighbors (search results), otherwise the graph would be broken
      if(result_queue.size() < edges_per_node) {
        fmt::print(stderr, "the graph search for the new node {} did only provided {} results \n", external_label, result_queue.size());
        perror("");
        abort();
      }

      // add an empty node to the graph (no neighbor information yet)
      const auto internal_index = this->graph_.addNode(external_label, add_task.feature.data());

      // for computing distances to neighbors not in the result queue
      const auto dist_func = this->graph_.getFeatureSpace().get_dist_func();
      const auto dist_func_param = this->graph_.getFeatureSpace().get_dist_func_param();
        
      // remove the worst edge of the good neighbors and connect them with this new node
      auto new_neighbors = std::vector<std::pair<uint32_t, float>>();
      while(new_neighbors.size() < edges_per_node) {
        auto result = result_queue.front();
        result_queue.pop();

        // check if the node is already in the edge list of the new node (added during a previous loop-run)
        // since all edges are undirected and the edge information of the new node does not yet exist, we search the other way around.
        if(this->graph_.hasEdge(result.getInternalIndex(), internal_index))
          continue;

        // find the worst edge of the new neighbor
        uint32_t bad_neighbor_index = 0;
        float bad_neighbor_weight = 0;
        const auto neighbor_weights = this->graph_.getNeighborWeights(result.getInternalIndex());
        const auto neighbor_indizies = this->graph_.getNeighborIndizies(result.getInternalIndex());
        for (size_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {
          const auto neighbor_index = neighbor_indizies[edge_idx];
          const auto neighbor_weight = neighbor_weights[edge_idx];

          // the new node might have been added to the neighbor list of this result-node during a previous loop
          // the suggest neighbors might already be in the edge list of the new node
          // the weight of the neighbor might not be worst than the current worst one
          if(internal_index != neighbor_index && this->graph_.hasEdge(neighbor_index, internal_index) == false && bad_neighbor_weight < neighbor_weight) {
            bad_neighbor_index = neighbor_index;
            bad_neighbor_weight = neighbor_weight;
          }          
        }

        // this should not be possible, otherwise the new node is connected to every node in the neighbor-list of the result-node and still has space for more
        if(bad_neighbor_weight == 0) {
          fmt::print(stderr, "it was not possible to find a bad edge in the neighbor list of node {} which would connect to node {} \n", result.getInternalIndex(), internal_index);
          perror("");
          abort();
        }

        // place the new node in the edge list of the result-node
        this->graph_.changeEdge(result.getInternalIndex(), bad_neighbor_index, internal_index, result.getDistance());
        new_neighbors.emplace_back(result.getInternalIndex(), result.getDistance());

        // place the new node in the edge list of the result-node and its worst edge neighbor
        const auto distance = dist_func(add_task.feature.data(), this->graph_.getFeatureVector(bad_neighbor_index), dist_func_param);
        this->graph_.changeEdge(bad_neighbor_index, result.getInternalIndex(), internal_index, distance);
        new_neighbors.emplace_back(bad_neighbor_index, distance);
      }

      // sort the neighbors by their neighbor indizies and store them in the new node
      std::sort(new_neighbors.begin(), new_neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
      auto neighbor_indizies = std::vector<uint32_t>();
      auto neighbor_weights = std::vector<float>();
      for (auto &&neighbor : new_neighbors) {
        neighbor_indizies.emplace_back(neighbor.first);
        neighbor_weights.emplace_back(neighbor.second);
      }
      this->graph_.changeEdges(internal_index, neighbor_indizies.data(), neighbor_weights.data());
    }

    /**
     * Removing a node from the graph.
     */
    void shrinkGraph(BuilderRemoveTask del_task) {
      fmt::print(stderr, "shrinking the graph by node {} is not yet implemented \n", del_task.label);
      perror("");
      abort();
    }

    bool improve() {
      return false;
    }


  public:

    /**
     * Build the graph. This could be run on a separate thread in an infinite loop.
     */
    auto& build(std::function<void(deglib::builder::BuilderStatus&)> callback, const bool infinite = false) {
      auto status = BuilderStatus{};
      const auto edge_per_node = this->graph_.getEdgesPerNode();

      // try to build an initial graph, containing the minium amount of nodes (edge_per_node + 1)
      if(graph_.size() < edge_per_node + 1) {

        // graph should be empty to initialize
        if(this->graph_.size() > 0) {
          fmt::print(stderr, "graph has already {} nodes and can therefore not be initialized \n", this->graph_.size());
          perror("");
          abort();
        }

        // wait until enough new entries exists to build the initial graph
        while(new_entry_queue_.size() < edge_per_node + 1)
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        // setup the initial graph
        auto initial_entries = std::vector<BuilderAddTask>();
        while(initial_entries.size() < edge_per_node + 1) {
          auto entry = this->new_entry_queue_.front();
          initial_entries.emplace_back(entry.label, std::move(entry.feature));
          this->new_entry_queue_.pop();
        }
        initialGraph(std::move(initial_entries));

        // inform the callback about the initial graph
        status.added += edge_per_node + 1;
        callback(status);
      }

      // run a loop to add, delete and improve the graph
      do{

        // add or delete a node
        if(this->new_entry_queue_.size() > 0 || this->remove_entry_queue_.size() > 0) {
          auto add_task = deglib::builder::BuilderAddTask{};
          auto del_task = deglib::builder::BuilderRemoveTask{};

          if(this->new_entry_queue_.size() > 0) 
            add_task = this->new_entry_queue_.front();

          if(this->remove_entry_queue_.size() > 0) 
            del_task = this->remove_entry_queue_.front();

          if(add_task.timestamp < del_task.timestamp) {
            extendGraph(add_task);
            status.added++;
            this->new_entry_queue_.pop();
          } else {
            shrinkGraph(del_task);
            status.deleted++;
            this->remove_entry_queue_.pop();
          }
        }

        // try to improve the graph
        for (size_t swap_try = 0; swap_try < this->swap_tries_; swap_try++) {
          status.tries++;

          if(this->improve()) {
						status.improved++;
						swap_try -= this->additional_swap_tries_;
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