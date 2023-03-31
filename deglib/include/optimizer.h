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
#include "analysis.h"

namespace deglib::builder
{

struct RNGTriple {
  uint32_t index0;
  uint32_t index1;
  uint32_t index2;
  float weights;
};

class EvenRegularGraphOptimizer {

    std::mt19937& rnd_;
    deglib::graph::MutableGraph& graph_;


  public:

    EvenRegularGraphOptimizer(deglib::graph::MutableGraph& graph, std::mt19937& rnd) 
      : graph_(graph), rnd_(rnd) {
    }

  private:
  

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


  public:

    /**
     * Does not produce better graphs
     */
    auto& removeNonRngEdges() {
      auto start = std::chrono::system_clock::now();

      auto& graph = this->graph_;
      const auto vertex_count = graph.size();
      const auto edge_per_vertex =graph.getEdgesPerNode();

      uint32_t removed_rng_edges = 0;
      for (uint32_t i = 0; i < vertex_count; i++) {
        const auto vertex_index = i;

        const auto neighbor_indices = graph.getNeighborIndices(vertex_index);
        const auto neighbor_weights = graph.getNeighborWeights(vertex_index);

        // find all none rng conform neighbors
        for (uint32_t n = 0; n < edge_per_vertex; n++) {
          const auto neighbor_index = neighbor_indices[n];
          const auto neighbor_weight = neighbor_weights[n];

          if(deglib::analysis::checkRNG(graph, edge_per_vertex, vertex_index, neighbor_index, neighbor_weight) == false) {
            //fmt::print("non-RNG between {} and {}\n", vertex_index, neighbor_index);
            graph.changeEdge(vertex_index, neighbor_index, vertex_index, 0);
            graph.changeEdge(neighbor_index, vertex_index, neighbor_index, 0);
            removed_rng_edges++;
          }
        }
      }

      auto duration_ms = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count());
      auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph);
      auto valid_weights = deglib::analysis::check_graph_weights(graph);
      auto connected = deglib::analysis::check_graph_connectivity(graph);
      auto duration = duration_ms / 1000;
      fmt::print("{:7} vertices, removed {:7} edges, {:5}s, improv, Q: {:4.2f}, {} connected & {}\n", 
                vertex_count, removed_rng_edges, duration, avg_edge_weight, connected ? "" : "not", valid_weights ? "valid" : "invalid");
      start = std::chrono::system_clock::now();
            
      return this->graph_;
    }

    /**
     * Does not produce better graphs
     */
    auto& reduceNonRngEdges() {
      auto start = std::chrono::system_clock::now();
      uint64_t duration_ms = 0;

      auto& graph = this->graph_;
      const auto vertex_count = graph.size();
      const auto edge_per_vertex =graph.getEdgesPerNode();

      uint32_t removed_triple = 0;
      auto noneRNGConformNeighborIndices = std::vector<uint32_t>();
      auto noneRNGConformNeighbors = std::vector<RNGTriple>();
      for (uint32_t i = 0; i < vertex_count; i++) {
        const auto vertex_index = i;

        const auto neighbor_indices = graph.getNeighborIndices(vertex_index);
        const auto neighbor_weights = graph.getNeighborWeights(vertex_index);

        // find all none rng conform neighbors
        noneRNGConformNeighborIndices.clear();
        for (uint32_t n = 0; n < edge_per_vertex; n++) {
          const auto neighbor_index = neighbor_indices[n];
          const auto neighbor_weight = neighbor_weights[n];
          if(deglib::analysis::checkRNG(graph, edge_per_vertex, vertex_index, neighbor_index, neighbor_weight) == false) {
            //fmt::print("Found a none rng-conform edge between {} {}\n", vertex_index, neighbor_index);
            noneRNGConformNeighborIndices.emplace_back(neighbor_index);
          }
        }

        // of those neighbors find thoese pairs which are adjacent to each other with another none conform RNG edge
        noneRNGConformNeighbors.clear();
        for (size_t n1 = 0; n1 < noneRNGConformNeighborIndices.size(); n1++) {
          const auto neighbor_index1 = noneRNGConformNeighborIndices[n1];
          for (size_t n2 = n1; n2 < noneRNGConformNeighborIndices.size(); n2++) {
            const auto neighbor_index2 = noneRNGConformNeighborIndices[n2];
            const auto neighbors_weight = graph.getEdgeWeight(neighbor_index1, neighbor_index2);
            if(neighbors_weight >= 0 && deglib::analysis::checkRNG(graph, edge_per_vertex, neighbor_index1, neighbor_index2, neighbors_weight) == false) {
              //fmt::print("Found three none rng-conform edges between three vertices: {} {} {}\n", vertex_index, neighbor_index1, neighbor_index2);

              const auto weight_sum = neighbors_weight + graph.getEdgeWeight(vertex_index, neighbor_index1) + graph.getEdgeWeight(vertex_index, neighbor_index2);
              noneRNGConformNeighbors.emplace_back(vertex_index, neighbor_index1, neighbor_index2, weight_sum);
            }
          }
        }

        if(noneRNGConformNeighbors.size() > 0) {
          std::sort(noneRNGConformNeighbors.begin(), noneRNGConformNeighbors.end(), [](const auto& x, const auto& y){return x.weights > y.weights;});
          const auto triple = noneRNGConformNeighbors[0];
          
          graph.changeEdge(triple.index0, triple.index1, triple.index0, 0);
          graph.changeEdge(triple.index0, triple.index2, triple.index0, 0);

          graph.changeEdge(triple.index1, triple.index0, triple.index1, 0);
          graph.changeEdge(triple.index1, triple.index2, triple.index1, 0);

          graph.changeEdge(triple.index2, triple.index0, triple.index2, 0);
          graph.changeEdge(triple.index2, triple.index1, triple.index2, 0);
          removed_triple++;
        }

        if(i % 100000 == 0) {
          duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count());
          auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph);
          auto valid_weights = deglib::analysis::check_graph_weights(graph);
          auto connected = deglib::analysis::check_graph_connectivity(graph);
          auto duration = duration_ms / 1000;
          fmt::print("{:7} vertices, removed {:7} triple edges, {:5}s, improv, Q: {:4.2f}, {} connected & {}\n", 
                    i, removed_triple, duration, avg_edge_weight, connected ? "" : "not", valid_weights ? "valid" : "invalid");
          start = std::chrono::system_clock::now();
        }
      }
      
      return this->graph_;
    }

};

} // end namespace deglib::builder