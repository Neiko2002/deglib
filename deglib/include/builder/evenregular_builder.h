#pragma once

#include <random>

#include "graph.h"

namespace deglib::builder
{

class EvenRegularGraphBuilder {

    const uint32_t extend_k_;         // k value for extending the graph
    const uint32_t improve_k_;        // k value for improving the graph
    const uint32_t max_path_length_;  // max amount of changes before canceling an improvement try

    std::mt19937& rnd_;
    deglib::graph::MutableGraph& graph_;

    std::queue<std::pair<uint32_t, const std::byte*>> new_entry_queue_;

  public:

    EvenRegularGraphBuilder(deglib::graph::MutableGraph& graph, const uint32_t extend_k, const uint32_t improve_k, const uint32_t max_path_length, std::mt19937& rnd) 
      : graph_(graph), extend_k_(extend_k), improve_k_(improve_k), max_path_length_(max_path_length), rnd_(rnd) {
    }

    /**
     * Presents the builder a new entry which it will append to the graph in the build() process.
     */ 
    void addEntry(uint32_t label, const std::byte* feature) {
      new_entry_queue_.emplace(label, feature);
    }


  private:
  
    auto size() {
      return new_entry_queue_.size();
    }




  public:

    /**
     * Build the graph. This could be run on a separate thread in an infinite loop.
     */
    auto build() {
      return size();
    }

};

} // end namespace deglib::builder