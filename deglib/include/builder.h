#pragma once

#include <random>
#include <chrono>
#include <thread>

#include <fmt/core.h>

#include "graph.h"

namespace deglib::builder
{

struct BuilderEntry {
  uint32_t label;
  std::vector<std::byte> feature;
};

class EvenRegularGraphBuilder {

    const uint32_t extend_k_;         // k value for extending the graph
    const uint32_t improve_k_;        // k value for improving the graph
    const uint32_t max_path_length_;  // max amount of changes before canceling an improvement try

    std::mt19937& rnd_;
    deglib::graph::MutableGraph& graph_;

    std::queue<BuilderEntry> new_entry_queue_;

    // should the build loop run until the stop method is called
    bool infinite_;

  public:

    EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, const uint32_t extend_k, const uint32_t improve_k, const uint32_t max_path_length, std::mt19937& rnd) 
      : graph_(graph), extend_k_(extend_k), improve_k_(improve_k), max_path_length_(max_path_length), rnd_(rnd) {
    }

    /**
     * Presents the builder a new entry which it will append to the graph in the build() process.
     */ 
    void addEntry(uint32_t label, std::vector<std::byte> feature) {
      new_entry_queue_.emplace(label, std::move(feature));
    }


  private:
  
    auto size() {
      return this->new_entry_queue_.size();
    }

    /**
     * The initial graph contains of "edges-per-node + 1" nodes.
     * Every node in this graph is connected to all other nodes.
     */
    void initialGraph(const std::vector<BuilderEntry> entries) {
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
      auto neighbor_indizies = std::vector<uint32_t>(edges_per_node);
      auto neighbor_weights = std::vector<float>(edges_per_node);
      for (auto &&entry : entries)   
        this->graph_.addNode(entry.label, entry.feature.data(), neighbor_indizies.data(), neighbor_weights.data());
      
      // setup the edges for every node
      auto neighbors = std::vector<std::pair<uint32_t,float>>();
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


  public:

    /**
     * Build the graph. This could be run on a separate thread in an infinite loop.
     */
    auto& build(std::function<void(uint64_t,uint64_t,uint64_t,uint64_t)> callback, bool infinite = false) {
      uint64_t step = 0;
      uint64_t added = 0;
      uint64_t deleted = 0;
      uint64_t improved = 0;
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
        auto initial_entries = std::vector<BuilderEntry>();
        while(initial_entries.size() < edge_per_node + 1) {
          auto entry = this->new_entry_queue_.front();
          initial_entries.emplace_back(entry.label, std::move(entry.feature));
          this->new_entry_queue_.pop();
        }
        initialGraph(std::move(initial_entries));

        // inform the callback about the initial graph
        added += edge_per_node + 1;
        callback(step, added, deleted, improved);
      }

      // run a loop to add, delete and improve the graph
      this->infinite_ = infinite;
      do{

        callback(++step, added, deleted, improved);
      }
      while(infinite_);

      // return the finished graph
      return this->graph_;
    }

    /**
     * Stop the build process
     */
    void stop() {
      this->infinite_ = false;
    }
};

} // end namespace deglib::builder