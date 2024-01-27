#pragma once

#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <functional>
#include <span>
#include <memory>
#include <array>

#include <fmt/core.h>
#include <fmt/format.h>

#include "graph.h"

#include <unordered_set>
#include <unordered_map>
#include <tsl/robin_set.h>

namespace deglib::builder
{
class EvenRegularGraphBuilderExperimental {

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

    std::atomic<uint64_t> manipulation_counter_;
    std::deque<BuilderAddTask> new_entry_queue_;
    std::queue<BuilderRemoveTask> remove_entry_queue_;

    // should the build loop run until the stop method is called
    bool stop_building_ = false;
    const float rng_factor_ = 1.5f;

    bool rng_add_ = true;
    bool rng_add_minimal_ = false;
    bool rng_swap_ = true;
    bool rng_swap_minimal_ = true;
    bool rng_swap_step0_ = true;

    // compute the most median entry vertex for a good nearest neighbor search
    std::vector<float> sum_fv_;
    std::vector<uint32_t> entry_vertex_indices_;

    uint64_t hop_sum_ = 0;
    uint64_t dist_cal_sum_ = 0;
    uint64_t checked_vertices_sum_ = 0;
    uint64_t search_count_ = 0;

  public:
    EvenRegularGraphBuilderExperimental(deglib::graph::MutableGraph& graph, std::mt19937& rnd, 
                            const uint8_t extend_k, const float extend_eps, const bool extend_highLID, 
                            const uint8_t improve_k, const float improve_eps, const bool improve_highLID, const uint8_t improve_step_factor = 2,
                            const uint8_t max_path_length = 10, const uint32_t swap_tries = 3, const uint32_t additional_swap_tries = 3) 
      : graph_(graph), rnd_(rnd), extend_k_(extend_k), extend_eps_(extend_eps), extend_highLID_(extend_highLID), 
        improve_k_(improve_k), improve_eps_(improve_eps), improve_highLID_(improve_highLID), improve_step_factor_(improve_step_factor),
        max_path_length_(max_path_length), swap_tries_(swap_tries), additional_swap_tries_(additional_swap_tries) {

        // TODO store information in graph
        // compute the median vertex vertex for a good nearest neighbor search
        entry_vertex_indices_ = std::vector<uint32_t> { 0 };
        const auto feature_dims = graph.getFeatureSpace().dim();
        const auto graph_size = (uint32_t) graph.size();
        sum_fv_ = std::vector<float>(feature_dims);
        for (uint32_t i = 0; i < graph_size; i++) {
            auto fv = reinterpret_cast<const float*>(graph.getFeatureVector(i));
            for (size_t dim = 0; dim < feature_dims; dim++) 
                sum_fv_[dim] += fv[dim];
        }
        updateEntryNode();
    }

    EvenRegularGraphBuilderExperimental(deglib::graph::MutableGraph& graph, std::mt19937& rnd, const uint32_t swaps) 
      : EvenRegularGraphBuilderExperimental(graph, rnd, 
                                graph.getEdgesPerNode(), 0.2f, true,
                                graph.getEdgesPerNode(), 0.02f, false,
                                2, 10, swaps, swaps) {
    }

    EvenRegularGraphBuilderExperimental(deglib::graph::MutableGraph& graph, std::mt19937& rnd) 
      : EvenRegularGraphBuilderExperimental(graph, rnd, 1) {
    }

    void clearSearchStats() {
      hop_sum_ = 0;
      dist_cal_sum_ = 0;
      checked_vertices_sum_ = 0;
      search_count_ = 0;
    }

    double getAvgHops() const {
      return ((double)hop_sum_) / search_count_;
    }

    double getAvgDistCalcs() const {
      return ((double)dist_cal_sum_) / search_count_;
    }

     double getAvgCheckedVertices() const {
      return ((double)checked_vertices_sum_) / search_count_;
    }

    /**
     * Provide the builder a new entry which it will append to the graph in the build() process.
     */ 
    void addEntry(const uint32_t label, std::vector<std::byte> feature) {
      auto manipulation_index = manipulation_counter_.fetch_add(1);
      new_entry_queue_.emplace_back(label, manipulation_index, std::move(feature));
    }

    /**
     * Command the builder to remove a vertex from the graph as fast as possible.
     */ 
    void removeEntry(const uint32_t label) {
      auto manipulation_index = manipulation_counter_.fetch_add(1);
      remove_entry_queue_.emplace(label, manipulation_index);
    }

  private:

      void addFeatureToMean(const std::byte* new_feature) {
      const auto feature = reinterpret_cast<const float*>(new_feature);
      const auto feature_dims = graph_.getFeatureSpace().dim();
      for (size_t dim = 0; dim < feature_dims; dim++) 
        sum_fv_[dim] += feature[dim];

      if(graph_.size() % 1000 == 0) 
        updateEntryNode();
    }

    void removeFeatureFromMean(const std::byte* new_feature) {
      const auto feature = reinterpret_cast<const float*>(new_feature);
      const auto feature_dims = graph_.getFeatureSpace().dim();
      for (size_t dim = 0; dim < feature_dims; dim++) 
        sum_fv_[dim] -= feature[dim];

      if(graph_.size() % 1000 == 0) 
        updateEntryNode();
    }

    void updateEntryNode() {
      if(graph_.size() > uint32_t(graph_.getEdgesPerNode() + 1)) {
        const auto feature_dims = graph_.getFeatureSpace().dim();
        const auto graph_size = graph_.size();
        auto avg_fv = std::vector<float>(feature_dims);
        for (size_t dim = 0; dim < feature_dims; dim++) 
          avg_fv[dim] = sum_fv_[dim] / graph_size;

        const auto seed = std::vector<uint32_t> { entry_vertex_indices_[0] };
        auto result_queue = graph_.search(seed, reinterpret_cast<const std::byte*>(avg_fv.data()), extend_eps_, extend_k_);
        entry_vertex_indices_[0] = result_queue.top().getInternalIndex();

        this->hop_sum_ += result_queue.hop_sum_;
        this->dist_cal_sum_ += result_queue.dist_cal_sum_;
        this->checked_vertices_sum_ += result_queue.checked_vertices_sum_;
        this->search_count_++;
      }
    }
  
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
     * The initial graph contains of "edges-per-vertex + 1" vertices.
     * Every vertex in this graph is connected to all other vertices.
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

      // setup the vertices first, to be able to get their internal indices
      for (auto &&entry : entries)   
        graph.addNode(entry.label, entry.feature.data());
      
      // setup the edges for every vertex
      const auto edges_per_vertex = graph.getEdgesPerNode();
      auto neighbors = std::vector<std::pair<uint32_t,float>>();
      auto neighbor_indices = std::vector<uint32_t>(edges_per_vertex);
      auto neighbor_weights = std::vector<float>(edges_per_vertex);
      for (size_t entry_idx = 0; entry_idx < size; entry_idx++) {

        // gather the edge distances to the other vertices
        neighbors.clear();
        for (size_t i = 0; i < size; i++) {

          // skip the vertex to which we collect the edges for
          if(i == entry_idx) continue;

          const auto neighbor_index = graph.getInternalIndex(entries[i].label);
          const auto distance = matrix[entry_idx][i];
          neighbors.emplace_back(neighbor_index, distance);
        }
        
        // sort the edges by their internal index values
        std::sort(neighbors.begin(), neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
        neighbor_indices.clear();
        neighbor_weights.clear();
        for (auto &&neighbor : neighbors) {
          neighbor_indices.emplace_back(neighbor.first);
          neighbor_weights.emplace_back(neighbor.second);
        }
        
        // store the edges of the new vertex
        const auto internal_index = graph.getInternalIndex(entries[entry_idx].label);
        graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());
      }
    }

    /**
     * Extend the graph with a new vertex. Find good existing vertex to which this new vertex gets connected.
     */
    void extendGraph(const BuilderAddTask& add_task) {
      auto& graph = this->graph_;
      const auto external_label = add_task.label;

      // graph should not contain a vertex with the same label
      if(graph.hasNode(external_label)) {
        fmt::print(stderr, "graph contains vertex {} already. can not add it again \n", external_label);
        perror("");
        abort();
      }

      // fully connect all vertices
      const auto new_vertex_feature = add_task.feature.data();
      const auto edges_per_vertex = uint32_t(graph.getEdgesPerNode());
      if(graph.size() < edges_per_vertex+1) {

        // add an empty vertex to the graph (no neighbor information yet)
        //addFeatureToMean(new_vertex_feature);
        const auto internal_index = graph.addNode(external_label, new_vertex_feature);

        // connect the new vertex to all other vertices in the graph
        const auto dist_func = graph.getFeatureSpace().get_dist_func();
        const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
        for (size_t i = 0; i < graph.size(); i++) {
          if(i != internal_index) {
            const auto dist = dist_func(new_vertex_feature, graph.getFeatureVector(i), dist_func_param);
            graph.changeEdge(i, i, internal_index, dist);
            graph.changeEdge(internal_index, internal_index, i, dist);
          }
        }

        return;
      }

      // find good neighbors for the new vertex
      auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
      const std::vector<uint32_t> entry_vertex_indices = { distrib(this->rnd_) };
      auto top_list = graph.search(entry_vertex_indices, new_vertex_feature, this->extend_eps_, std::max(uint32_t(this->extend_k_), edges_per_vertex));
      const auto results = topListAscending(top_list);

      this->hop_sum_ += top_list.hop_sum_;
      this->dist_cal_sum_ += top_list.dist_cal_sum_;
      this->checked_vertices_sum_ += top_list.checked_vertices_sum_;
      this->search_count_++;

      // their should always be enough neighbors (search results), otherwise the graph would be broken
      if(results.size() < edges_per_vertex) {
        fmt::print(stderr, "the graph search for the new vertex {} did only provided {} results \n", external_label, results.size());
        perror("");
        abort();
      }

      // add an empty vertex to the graph (no neighbor information yet)
      // addFeatureToMean(new_vertex_feature);
      const auto internal_index = graph.addNode(external_label, new_vertex_feature);

      // for computing distances to neighbors not in the result queue
      const auto dist_func = graph.getFeatureSpace().get_dist_func();
      const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
     
      // adding neighbors happens in two phases, the first tries to retain RNG, the second adds them without checking
      int check_rng_phase = 1; // 1 = activated, 2 = deactived

      // remove an edge of the good neighbors and connect them with this new vertex
      auto new_neighbors = std::vector<std::pair<uint32_t, float>>();
      while(new_neighbors.size() < edges_per_vertex) {
        for (size_t i = 0; i < results.size() && new_neighbors.size() < edges_per_vertex; i++) {
          const auto candidate_index = results[i].getInternalIndex();
          const auto candidate_weight = results[i].getDistance();
          const auto& result = results[i];

          // check if the vertex is already in the edge list of the new vertex (added during a previous loop-run)
          // since all edges are undirected and the edge information of the new vertex does not yet exist, we search the other way around.
          if(graph.hasEdge(result.getInternalIndex(), internal_index)) 
            continue;

          // does this vertex has a neighbor which is connected to the new vertex and has a lower distance?
          // if(check_rng_phase <= 1 && deglib::analysis::check_NSW_RNG(graph, edges_per_vertex, candidate_index, internal_index, candidate_weight) == false) 
          // if(check_rng_phase <= 1 && deglib::analysis::check_SSG_RNG(graph, candidate_index, candidate_weight, 60, new_neighbors) == false) 
          if(check_rng_phase <= 1 && deglib::analysis::checkRNG(graph, edges_per_vertex, candidate_index, internal_index, candidate_weight) == false) 
            continue;

          // This version is good for high LID datasets or small graphs with low distance count limit during ANNS
          uint32_t new_neighbor_index = 0;
          float new_neighbor_distance = -1;
          if(extend_highLID_) {

            // find the worst edge of the new neighbor
            // float new_neighbor_weight =  std::numeric_limits<float>::max();   // version B in the paper
            float new_neighbor_weight = -1;                                      // version C in the paper
            float new_neighbor_weight_orig = -1;
            const auto neighbor_weights = graph.getNeighborWeights(result.getInternalIndex());
            const auto neighbor_indices = graph.getNeighborIndices(result.getInternalIndex());

            // float avg_weight = 0;
            // float sum2_weight = 0;
            // for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
            //   const auto w = neighbor_weights[edge_idx];
            //   avg_weight += w;
            //   sum2_weight += w*w;
            // }
            // avg_weight /= edges_per_vertex;
            // float avg_variance = std::sqrt(sum2_weight/edges_per_vertex - avg_weight*avg_weight);

            for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
              const auto neighbor_index = neighbor_indices[edge_idx];
              //const auto neighbor_weight = neighbor_weights[edge_idx];

              // the suggested neighbor might already be in the edge list of the new vertex
              if(graph.hasEdge(neighbor_index, internal_index))
                continue;

              // // new edge between new vertex and neighbor would not be RNG conform if the old edge between candidate and neighbor get not deleted
              // // new edge is therefore not a good edge -------> since candidate is the closest to the new vertex, most of its neighbor would not fullfill this condition
              // if(new_vertex_neighbor_distance > std::max(candidate_weight, neighbor_weights[edge_idx]))
              //   continue;

              // find a non-RNG edge between the candidate_index and neighbor_index, which would be RNG conform between internal_index and neighbor_index
              auto factor = 1.0f;
              // if(check_rng_phase <= 2 && deglib::analysis::checkRNG(graph, internal_index, neighbor_index, new_neighbors) == false)
              //   continue;
              // if(check_rng_phase <= 2 && deglib::analysis::checkRNG(graph, edges_per_vertex, candidate_index, neighbor_index, neighbor_weights[edge_idx]))
              //   continue;
              // if(deglib::analysis::checkRNG(graph, edges_per_vertex, candidate_index, neighbor_index, neighbor_weights[edge_idx]) == false)
                // factor *= rng_factor_;
                // factor = avg_variance;
              //   // factor *= neighbor_weights[edge_idx];
              //   // factor = avg_variance * std::max(1.0f, neighbor_weights[edge_idx]/avg_weight);
              // // if(deglib::analysis::checkRNG(graph, internal_index, neighbor_index, new_neighbors))
              // //   factor *= rng_factor_/2;

              // const auto neighbor_weight = neighbor_weights[edge_idx] * factor;
              const auto neighbor_weight = neighbor_weights[edge_idx] + factor;

              // the suggested neighbor might already be in the edge list of the new vertex
              // the weight of the neighbor might not be worst than the current worst one
              // if(neighbor_weight < new_neighbor_weight) {       // version B in the paper              
              if(neighbor_weight > new_neighbor_weight) {       // version C in the paper

                // rng approximation
                // if(check_rng && rng_add_ && rng_add_minimal_ == false) {
                //   const auto neighbor_distance = dist_func(new_vertex_feature, graph.getFeatureVector(neighbor_index), dist_func_param);
                //   if(checkRNG(graph, edges_per_vertex, neighbor_index, internal_index, neighbor_distance) == false) 
                //     continue;
                // }

                new_neighbor_weight = neighbor_weight;
                new_neighbor_weight_orig = neighbor_weights[edge_idx];
                new_neighbor_index = neighbor_index;
              }       
            }

                        // new_neighbor_weight == -1 should not be possible, otherwise the new vertex is connected to every vertex in the neighbor-list of the result-vertex and still has space for more
            if(new_neighbor_weight == -1) {
            // if(new_neighbor_weight == std::numeric_limits<float>::max()) {
              continue;
              // fmt::print(stderr, "it was not possible to find a bad edge in the neighbor list of vertex {} which would connect to vertex {} \n", candidate_index, internal_index);
              // perror("");
              // abort();
            }

            // we have choosen an edge between candidate_index and its neighbor which is currently not RNG conform
            // if(graph.size() > 10000 && deglib::analysis::checkRNG(graph, edges_per_vertex, candidate_index, new_neighbor_index, new_neighbor_weight_orig) == true) {
            //   fmt::print("\nFor new vertex {} find feighbors of {} with an average weight of {} and variance {}\n", internal_index, candidate_index, avg_weight, avg_variance);

            //   for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
            //     const auto neighbor_index = neighbor_indices[edge_idx];
            //     const auto neighbor_weight = neighbor_weights[edge_idx];

            //     // the suggested neighbor might already be in the edge list of the new vertex
            //     if(graph.hasEdge(neighbor_index, internal_index))
            //       continue;

            //     // find a non-RNG edge between the candidate_index and neighbor_index, which would be RNG conform between internal_index and neighbor_index
            //     auto isRNG = deglib::analysis::checkRNG(graph, edges_per_vertex, candidate_index, neighbor_index, neighbor_weight);
            //     auto boosted = neighbor_weight;
            //     if(isRNG == false)
            //       boosted = uint32_t(boosted * rng_factor_);

            //     auto f1 = 1.0f;
            //     if(isRNG == false)
            //       f1 *= std::max(1.0f, neighbor_weight/avg_weight);

            //     fmt::print("neighbor {:5}, weight {:7.0f}, boosted {:7.0f}{:1}, f1 {:0.4f}, rng {:5}\n", neighbor_index, neighbor_weight, boosted, (neighbor_index==new_neighbor_index) ? "*":"", f1, isRNG ? "true":"false");
            //   }
            // }

            // if(check_rng_phase <= 1 && deglib::analysis::check_SSG_RNG(graph, candidate_index, candidate_weight, 60, new_neighbors) == false) 
            //   continue;

            new_neighbor_distance = dist_func(new_vertex_feature, graph.getFeatureVector(new_neighbor_index), dist_func_param); 
          }
          else
          {
            // find the edge which improves the distortion the most: (distance_new_edge1 + distance_new_edge2) - distance_removed_edge
            {
              float best_distortion = std::numeric_limits<float>::max();
              const auto neighbor_indices = graph.getNeighborIndices(result.getInternalIndex());
              const auto neighbor_weights = graph.getNeighborWeights(result.getInternalIndex());
              for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
                const auto neighbor_index = neighbor_indices[edge_idx];
                if(graph.hasEdge(neighbor_index, internal_index) == false) {
                  const auto neighbor_distance = dist_func(new_vertex_feature, graph.getFeatureVector(neighbor_index), dist_func_param);

                  // rng approximation
                  // if(check_rng && rng_add_ && rng_add_minimal_ == false && checkRNG(graph, edges_per_vertex, neighbor_index, internal_index, neighbor_distance) == false)
                  //     continue;
                  
                  // take the neighbor with the best distance to the new vertex, which might already be in its edge list
                  // float distortion = neighbor_distance;   // version A in the paper
                  float distortion = (result.getDistance() + neighbor_distance) - neighbor_weights[edge_idx];   // version D in the paper
                  if(distortion < best_distortion) {
                    best_distortion = distortion;
                    new_neighbor_index = neighbor_index;
                    new_neighbor_distance = neighbor_distance;
                  }          
                }
              }
            }
          }

          // this should not be possible, otherwise the new vertex is connected to every vertex in the neighbor-list of the result-vertex and still has space for more
          if(new_neighbor_distance == -1) {
            continue;
            //fmt::print(stderr, "it was not possible to find a bad edge in the neighbor list of vertex {} which would connect to vertex {} \n", result.getInternalIndex(), internal_index);
            //perror("");
            //abort();
          }

          // place the new vertex in the edge list of the result-vertex
          graph.changeEdge(result.getInternalIndex(), new_neighbor_index, internal_index, result.getDistance());
          new_neighbors.emplace_back(result.getInternalIndex(), result.getDistance());

          // place the new vertex in the edge list of the best edge neighbor
          graph.changeEdge(new_neighbor_index, result.getInternalIndex(), internal_index, new_neighbor_distance);
          new_neighbors.emplace_back(new_neighbor_index, new_neighbor_distance);
        }
        check_rng_phase++;
      }

      if(new_neighbors.size() < edges_per_vertex) {
        fmt::print(stderr, "could find only {} good neighbors for the new vertex {} need {}\n", new_neighbors.size(), internal_index, edges_per_vertex);
        perror("");
        abort();
      }

      // sort the neighbors by their neighbor indices and store them in the new vertex
      {
        std::sort(new_neighbors.begin(), new_neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
        auto neighbor_indices = std::vector<uint32_t>(new_neighbors.size());
        auto neighbor_weights = std::vector<float>(new_neighbors.size());
        for (size_t i = 0; i < new_neighbors.size(); i++) {
          const auto& neighbor = new_neighbors[i];
          neighbor_indices[i] = neighbor.first;
          neighbor_weights[i] = neighbor.second;
        }
        graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());  
      }


      // try to improve some of the non-perfect edges (not part of the range-search)
      if(improve_k_ > 0) {
        auto nonperfect_neighbors = std::vector<BoostedEdge>();
        for (size_t i = 0; i < new_neighbors.size(); i++) {
          const auto& neighbor = new_neighbors[i];

          // was the new neighbor found by the range-search or is just a neighbor of a neighbor
          bool perfect = false;
          for (size_t r = 0; r < results.size(); r++) {
            const auto& result = results[r];
            if(result.getInternalIndex() == neighbor.first) {
              perfect = true;
              break;
            }
          } 

          if(perfect == false && graph.hasEdge(internal_index, neighbor.first)) {
            // bool rng = deglib::analysis::check_SSG_RNG(graph, neighbor.first, neighbor.second, 60, new_neighbors);
            bool rng = deglib::analysis::checkRNG(graph, edges_per_vertex, internal_index, neighbor.first, neighbor.second);
            // bool rng = deglib::analysis::check_NSW_RNG(graph, edges_per_vertex, internal_index, neighbor.first, neighbor.second);
            nonperfect_neighbors.emplace_back(internal_index, neighbor.first, neighbor.second, neighbor.second, rng);
            // nonperfect_neighbors.emplace_back(neighbor.first, neighbor.second, neighbor.second * (rng ? 1.0f : rng_factor_), rng);
          }
        }

        std::sort(nonperfect_neighbors.begin(), nonperfect_neighbors.end(), [](const auto& x, const auto& y){return x.boost < y.boost;}); // low to high
        for (size_t i = 0; i < nonperfect_neighbors.size(); i++) {
          // if(nonperfect_neighbors[i].rng == false) { // none rng 
          // if(graph.hasEdge(internal_index, nonperfect_neighbors[i].to_vertex)) { // slow
          if(graph.hasEdge(internal_index, nonperfect_neighbors[i].to_vertex) && (i % 2 == 0)) { // normal
          // if(graph.hasEdge(internal_index, nonperfect_neighbors[i].to_vertex) && nonperfect_neighbors[i].rng == false) { // fast
          // if(graph.hasEdge(internal_index, nonperfect_neighbors[i].to_vertex) && (i < nonperfect_neighbors.size() / 2)) { // normal            
          // if(graph.hasEdge(internal_index, nonperfect_neighbors[i].to_vertex) && (i >= nonperfect_neighbors.size() / 2)) {
          // if(graph.hasEdge(internal_index, nonperfect_neighbors[i].to_vertex) && (nonperfect_neighbors[i].rng == false || i < nonperfect_neighbors.size() / 2)) {
          // if(graph.hasEdge(internal_index, nonperfect_neighbors[i].to_vertex) && (nonperfect_neighbors[i].rng == false || i >= nonperfect_neighbors.size() / 2)) {
            improveEdges(internal_index, nonperfect_neighbors[i].to_vertex, nonperfect_neighbors[i].weight); 
          }
        }
      }
    }

    /**
     * Removing a vertex from the graph.
     */
    void shrinkGraph(const BuilderRemoveTask& del_task) {
      auto& graph = this->graph_;
      const auto edges_per_vertex = std::min(graph.size(), uint32_t(graph.getEdgesPerNode()));
      
      // 1 remove the vertex and collect the vertices which are missing an edge
      const auto involved_indices = graph.removeNode(del_task.label);

      // 1.1 handle the use case where the graph does not have enough vertices to fulfill the edgesPerVertex requirement
		  //     and just remove the vertex without reconnecting the involved vertices because they are all fully connected
      if(graph.size() <= edges_per_vertex) 
        return;

      // 2 find pairs or groups of vertices which can reach each other		
		  auto unique_groups = std::unordered_set<std::shared_ptr<ReachableGroup>>();	
      {
        auto path_map = UnionFind(edges_per_vertex);
        auto reachable_groups = std::unordered_map<uint32_t, std::shared_ptr<ReachableGroup>>();	
        reachable_groups.reserve(edges_per_vertex);
        for (const auto involved_index : involved_indices) {
          reachable_groups.emplace(involved_index, std::make_shared<ReachableGroup>(involved_index, edges_per_vertex));
          path_map.Update(involved_index, involved_index);
        }

        // helper function to check if we need to find more connected components
        auto is_enough_free_connections = [](const std::vector<uint32_t>& vertices, const UnionFind& paths, const std::unordered_map<uint32_t, std::shared_ptr<ReachableGroup>>& groups) {
          size_t isolated_vertex_counter = 0;
          size_t available_connections_counter = 0;
          for(const auto& involved_vertex : vertices) {
            const auto reachable_Vertex = paths.Find(involved_vertex);
            if(involved_vertex == reachable_Vertex) {
              const auto& group = groups.at(reachable_Vertex);
              if(group->size() == 1)
                isolated_vertex_counter++;
              else if(group->getMissingEdgeSize() > 2)
                available_connections_counter += group->getMissingEdgeSize() - 2;
            }
          }
          return available_connections_counter < isolated_vertex_counter;
        };

        // 2.1 start with checking the adjacent neighbors
        size_t neighbor_check_depth = 0;
        auto check = std::unordered_set<uint32_t>(involved_indices.begin(), involved_indices.end());
        auto check_next = std::unordered_set<uint32_t>();
        while(is_enough_free_connections(involved_indices, path_map, reachable_groups)) {
          for(const auto check_vertex : check) {
            auto involved_vertex = path_map.Find(check_vertex);
            auto reachable_group = reachable_groups.at(involved_vertex);

            // check only involved vertices and vertices which can only reach 1 involved vertex
						// no need for big groups to find other groups at the expense of processing power
            if(neighbor_check_depth > 0 && reachable_group->size() > 1)
              continue;

            // check the neighbors of checkVertex if they can reach another reachableGroup
            auto neighbor_indices = graph.getNeighborIndices(check_vertex);
            for(uint32_t i = 0; i < edges_per_vertex; i++) {
              auto neighbor_index = neighbor_indices[i];

              // skip self references (loops)
              if(neighbor_index == check_vertex)
                continue;

              // which other involved vertex can be reached by this neighbor
              auto other_involved_vertex = path_map.Find(neighbor_index);

              // neighbor is not yet in the union find
              if(other_involved_vertex == path_map.getDefaultValue()) {
                path_map.Update(neighbor_index, involved_vertex);
                check_next.emplace(neighbor_index);
              }
              // the neighbor can reach another involved vertex
              else if(other_involved_vertex != involved_vertex) {
                path_map.Update(other_involved_vertex, involved_vertex);
                reachable_group->copyFrom(*reachable_groups.at(other_involved_vertex));
              }
            }
          }

          // prepare for the next iteration
          std::swap(check, check_next);
          check_next.clear();
          neighbor_check_depth++;
        }

        // copy the unique groups
        for (const auto involved_index : involved_indices) 
          unique_groups.emplace(reachable_groups.at(path_map.Find(involved_index)));
      }

      // 2.2 get all isolated vertices
      auto isolated_groups = std::unordered_set<std::shared_ptr<ReachableGroup>>();	
      for(const auto group : unique_groups)
        if(group->size() == 1)
          isolated_groups.emplace(group);

      // 2.3 find for every isolated vertex the best other involved vertex which is part of a unique group      
      auto new_edges = std::vector<BoostedEdge>();
      const auto& feature_space = graph.getFeatureSpace();
      const auto dist_func = feature_space.get_dist_func();
      const auto dist_func_param = feature_space.get_dist_func_param();
      for(const auto isolated_group : isolated_groups) {

        // are you still isolated?
        if(isolated_group->size() > 1)
          continue;

        const auto isolated_vertex = isolated_group->getVertexIndex();
        const auto isolated_vertex_feature = graph.getFeatureVector(isolated_vertex);

        // check the reachable groups for good candidates which can connect to the isolated vertex
        uint32_t best_candidate_index = 0;
        float best_candidate_distance = std::numeric_limits<float>::max();
        deglib::builder::ReachableGroup* best_candidate_group = nullptr;
        for (const auto candidate_group : unique_groups) {

          // skip all groups which do not have enough vertices missing an edge
          const auto& missing_edges = candidate_group->getMissingEdges();
          if(missing_edges.size() <= 2)
            continue;

          // find the candidate with the best distance to the isolated vertex
          for (const auto candidate : missing_edges) {
            const auto candidate_feature = graph.getFeatureVector(candidate);
            const auto distance = dist_func(isolated_vertex_feature, candidate_feature, dist_func_param);
            if(distance < best_candidate_distance) {
              best_candidate_distance = distance;
              best_candidate_index = candidate;
              best_candidate_group = candidate_group.get();
            }
          }
        }

        // found a good candidate, add the isolated vertex to its reachable group and an edge between them
        graph.changeEdge(isolated_vertex, isolated_vertex, best_candidate_index, best_candidate_distance);
        graph.changeEdge(best_candidate_index, best_candidate_index, isolated_vertex, best_candidate_distance);
        new_edges.emplace_back(isolated_vertex, best_candidate_index, best_candidate_distance, best_candidate_distance, true);

        // merge groups
        best_candidate_group->hasEdge(best_candidate_index);
        isolated_group->hasEdge(isolated_vertex);
        best_candidate_group->copyFrom(*isolated_group);

        unique_groups.erase(isolated_group);
      }

      // 3 reconnect the groups
      auto reachable_groups = std::vector(unique_groups.begin(), unique_groups.end());

      // Define a custom comparison function based on the size of the sets
      auto compareBySize = [](const std::shared_ptr<deglib::builder::ReachableGroup>& a, const std::shared_ptr<deglib::builder::ReachableGroup>& b) {
          return a->getMissingEdgeSize() < b->getMissingEdgeSize(); // < is ascending, > is descending
      };

      // Sort the groups by size in ascending order
      std::sort(reachable_groups.begin(), reachable_groups.end(), compareBySize);

      // 3.1 Find the biggest group and one of its vertices to one vertex of a smaller group. Repeat until only one group is left.
      while(reachable_groups.size() >= 2) {
        auto& reachable_group = *reachable_groups[reachable_groups.size()-1];
        auto& other_group = *reachable_groups[reachable_groups.size()-2];
        auto& reachable_vertices = reachable_group.getMissingEdges();
        auto& other_vertices = other_group.getMissingEdges();

        auto best_other_it = reachable_vertices.begin();
        auto best_reachable_it = reachable_vertices.begin();
        auto best_other_distance = std::numeric_limits<float>::max();

        // iterate over all its entries to find a vertex which is still missing an edge
        for(auto reachable_it = reachable_vertices.begin(); reachable_it != reachable_vertices.end(); ++reachable_it) {
          const auto reachable_index = *reachable_it;
          const auto reachable_feature = graph.getFeatureVector(reachable_index);

          // find another vertex in a smaller group, also missing an edge			
          // the other vertex and reachable_index can not share an edge yet, otherwise they would be in the same group due to step 2.1           
          for(auto other_it = other_vertices.begin(); other_it != other_vertices.end(); ++other_it) {
            const auto other_index = *other_it;
            const auto other_feature = graph.getFeatureVector(other_index);
            const auto candidate_dist = dist_func(reachable_feature, other_feature, dist_func_param);

            if(candidate_dist < best_other_distance) {
              best_other_it = other_it;
              best_reachable_it = reachable_it;
              best_other_distance = candidate_dist;
            }
          }
        }

        // connect reachable_index and other_index
        const auto reachable_index = *best_reachable_it;
        const auto other_index = *best_other_it;
        graph.changeEdge(reachable_index, reachable_index, other_index, best_other_distance);
        graph.changeEdge(other_index, other_index, reachable_index, best_other_distance);
        new_edges.emplace_back(other_index, reachable_index, best_other_distance, best_other_distance, false);

        // move the element from the list of missing edges
        reachable_group.hasEdge(reachable_index);
        other_group.hasEdge(other_index);

        // merge both groups
	      other_group.copyFrom(reachable_group);

        // remove the current group from the list of group since its merged
        reachable_groups.pop_back();
      }

      // 3.4 now all groups are reachable but still some vertices are missing edge, try to connect them to each other.
      auto remaining_indices = std::vector<uint32_t>(reachable_groups[0]->getMissingEdges().begin(), reachable_groups[0]->getMissingEdges().end());
      for (size_t i = 0; i < remaining_indices.size(); i++) {
        const auto index_A = remaining_indices[i];
        if(graph.hasEdge(index_A, index_A)) { // still missing an edge?

          // find a index_B with the smallest distance to index_A
          const auto feature_A = graph.getFeatureVector(index_A);
          auto best_index_B = -1;
          auto best_distance_AB = std::numeric_limits<float>::max();
          for (size_t j = i+1; j < remaining_indices.size(); j++) {
            const auto index_B = remaining_indices[j];
            if(graph.hasEdge(index_B, index_B) && graph.hasEdge(index_A, index_B) == false) {
              const auto new_neighbor_dist = dist_func(feature_A, graph.getFeatureVector(index_B), dist_func_param);
              if(new_neighbor_dist < best_distance_AB) {
                best_distance_AB = new_neighbor_dist;
                best_index_B = index_B;
              }
            }
          }

          // connect vertexA and vertexB
          if(best_index_B >= 0) {
            graph.changeEdge(index_A, index_A, best_index_B, best_distance_AB);
            graph.changeEdge(best_index_B, best_index_B, index_A, best_distance_AB);
            new_edges.emplace_back(best_index_B, index_A, best_distance_AB, best_distance_AB, false);
          }
        }
      }

      // 3.5 the remaining vertices can not be connected to any of the other involved vertices, because they already have an edge to all of them.
      for (size_t i = 0; i < remaining_indices.size(); i++) {
        const auto index_A = remaining_indices[i];
        if(graph.hasEdge(index_A, index_A)) { // still missing an edge?

          // scan the neighbors of the adjacent vertices of A and find a vertex B with the smallest distance to A
          const auto feature_A = graph.getFeatureVector(index_A);
          uint32_t best_index_B = 0;
          auto best_distance_AB = std::numeric_limits<float>::max();
          const auto neighbors_A = graph.getNeighborIndices(index_A);
          for (size_t n = 0; n < edges_per_vertex; n++) {
            const auto potential_indices = graph.getNeighborIndices(neighbors_A[n]);
            for (size_t p = 0; p < edges_per_vertex; p++) {
              const auto index_B = potential_indices[p];
              if(index_A != index_B && graph.hasEdge(index_A, index_B) == false) {
                const auto new_neighbor_dist = dist_func(feature_A, graph.getFeatureVector(index_B), dist_func_param);
                if(new_neighbor_dist < best_distance_AB) {
                  best_distance_AB = new_neighbor_dist;
                  best_index_B = index_B;
                }
              }
            }
          }

          // Get another vertex missing an edge called C and at this point sharing an edge with A (by definition of 3.2)
          for (size_t j = i+1; j < remaining_indices.size(); j++) {
            const auto index_C = remaining_indices[j];
            if(graph.hasEdge(index_C, index_C)) { // still missing an edge?
              const auto feature_C = graph.getFeatureVector(index_C);

              // check the neighborhood of B to find a vertex D not yet adjacent to C but with the smallest possible distance to C
              auto best_index_D = -1;
              auto best_distance_CD = std::numeric_limits<float>::max();
              const auto neighbors_B = graph.getNeighborIndices(best_index_B);
              for (size_t n = 0; n < edges_per_vertex; n++) {
                const auto index_D = neighbors_B[n];
                if(index_A != index_D && best_index_B != index_D && graph.hasEdge(index_C, index_D) == false) {
                  const auto new_neighbor_dist = dist_func(feature_C, graph.getFeatureVector(index_D), dist_func_param);
                  if(new_neighbor_dist < best_distance_CD) {
                    best_distance_CD = new_neighbor_dist;
                    best_index_D = index_D;
                  }
                }
              }

              // replace edge between B and D, with one between A and B as well as C and D
              graph.changeEdge(best_index_B, best_index_D, index_A, best_distance_AB);
              graph.changeEdge(index_A, index_A, best_index_B, best_distance_AB);
              graph.changeEdge(best_index_D, best_index_B, index_C, best_distance_CD);
              graph.changeEdge(index_C, index_C, best_index_D, best_distance_CD);
              new_edges.emplace_back(index_A, best_index_B, best_distance_AB, best_distance_AB, false);
              new_edges.emplace_back(index_C, best_index_D, best_distance_CD, best_distance_CD, false);

              break;
            }
          }
        }
      }

      // Define a custom comparison function based on the size of the sets
      auto compareByWeight = [](const BoostedEdge& a, const BoostedEdge& b) {
        return a.weight > b.weight; // < is ascending, > is descending
      };

      // Sort the groups by size in ascending order
      std::sort(new_edges.begin(), new_edges.end(), compareByWeight);

      // 4 try to improve some of the new edges
      for (size_t i = 0; i < new_edges.size(); i++) {
        const auto edge = new_edges[i];
        if(graph.hasEdge(edge.from_vertex, edge.to_vertex) && edge.rng)
        // if(graph.hasEdge(edge.from_vertex, edge.to_vertex) && (i < new_edges.size()/2))
        // if(graph.hasEdge(edge.from_vertex, edge.to_vertex) && (edge.rng || i % 2 == 0))
        // if(graph.hasEdge(edge.from_vertex, edge.to_vertex) && (edge.rng || i < new_edges.size()/2))
          improveEdges(edge.from_vertex, edge.to_vertex, edge.weight); 
      }
    }

    /**
     * Do not call this method directly instead call improve() to improve the graph.
     *  
     * This is the extended part of the optimization process.
     * The method takes an array where all graph changes will be documented.
	   * Node1 and vertex2 might be in a separate subgraph than vertex3 and vertex4.
     * Thru a series of edges swaps both subgraphs should be reconnected..
     * If those changes improve the graph this method returns true otherwise false. 
     * 
     * @return true if a good sequences of changes has been found
     */
    bool improveEdges(std::vector<deglib::builder::BuilderChange>& changes, uint32_t vertex1, uint32_t vertex2, uint32_t vertex3, uint32_t vertex4, float total_gain, const uint8_t steps) {
      auto& graph = this->graph_;
      const auto edges_per_vertex = graph.getEdgesPerNode();
      
      // the settings are the same for the first two iterations
      const auto high_variance_swaps = this->improve_highLID_;
      const auto search_eps = this->improve_eps_; 
      const auto search_k = this->improve_k_ - (uint8_t) std::max(0, (steps-1)*this->improve_step_factor_);

      if(high_variance_swaps) {
    
        // 1. Find a edge for vertex2 which connects to the subgraph of vertex3 and vertex4. 
        //    Consider only vertices of the approximate nearest neighbor search. Since the 
        //    search started from vertex3 and vertex4 all vertices in the result list are in 
        //    their subgraph and would therefore connect the two potential subgraphs.	
        float dist23 = 0;
        {
          const auto vertex2_feature = graph.getFeatureVector(vertex2);
          const std::vector<uint32_t> entry_vertex_indices = { vertex3, vertex4 };
          auto top_list = graph.search(entry_vertex_indices, vertex2_feature, search_eps, search_k);

          this->hop_sum_ += top_list.hop_sum_;
          this->dist_cal_sum_ += top_list.dist_cal_sum_;
          this->checked_vertices_sum_ += top_list.checked_vertices_sum_;
          this->search_count_++;

          // find a good new vertex3
          for(auto&& result : topListAscending(top_list)) {

            // TODO maybe making sure the new vertex3 is not the old vertex3 or even vertex4 helps
            if(vertex1 != result.getInternalIndex() && vertex2 != result.getInternalIndex() && graph.hasEdge(vertex2, result.getInternalIndex()) == false) {
              vertex3 = result.getInternalIndex();
              dist23 = result.getDistance();
              break;
            }
          }

          // no new vertex3 was found
          if(dist23 == 0)
            return false;

          // replace the temporary self-loop of vertex2 with a connection to vertex3. 
          graph.changeEdge(vertex2, vertex2, vertex3, dist23);
          changes.emplace_back(vertex2, vertex2, 0.f, vertex3, dist23);
          total_gain -= dist23;
        }

        // 2. All vertices are connected but the subgraph between vertex1/vertex2 and vertex3/vertex4 might just have one edge(vertex2, vertex3).
        //    Furthermore Node 3 has now to many edges, remove the worst one. Ignore the just added edge. 
        //    FYI: If the just selected vertex3 is the same as the old vertex3, this process might cut its connection to vertex4 again.
        //    This will be fixed in the next step or until the recursion reaches max_path_length.
        float dist34 = 0;
        {
          // 2.1 find the worst edge of vertex3
          uint32_t bad_neighbor_index = 0;
          float bad_neighbor_weight = 0.f;
          const auto neighbor_weights = graph.getNeighborWeights(vertex3);
          const auto neighbor_indices = graph.getNeighborIndices(vertex3);
          for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {

            // do not remove the edge which was just added
            if(neighbor_indices[edge_idx] != vertex2 && bad_neighbor_weight < neighbor_weights[edge_idx]) {
              bad_neighbor_index = neighbor_indices[edge_idx];
              bad_neighbor_weight = neighbor_weights[edge_idx];    
            }
          }

          // 2.2 Remove the worst edge of vertex3 to vertex4 and replace it with the connection to vertex2
          //     Add a temporary self-loop for vertex4 for the missing edge to vertex3
          vertex4 = bad_neighbor_index;
          dist34 = bad_neighbor_weight;
          total_gain += dist34;
          graph.changeEdge(vertex3, vertex4, vertex2, dist23);
          changes.emplace_back(vertex3, vertex4, dist34, vertex2, dist23);
          graph.changeEdge(vertex4, vertex3, vertex4, 0.f);
          changes.emplace_back(vertex4, vertex3, dist34, vertex4, 0.f);
        }

      }
      else
      {
      
        // 1. Find an edge for vertex2 which connects to the subgraph of vertex3 and vertex4. 
        //    Consider only vertices of the approximate nearest neighbor search. Since the 
        //    search started from vertex3 and vertex4 all vertices in the result list are in 
        //    their subgraph and would therefore connect the two potential subgraphs.	
        {
          const auto vertex2_feature = graph.getFeatureVector(vertex2);
          const std::vector<uint32_t> entry_vertex_indices = { vertex3, vertex4 };
          auto top_list = graph.search(entry_vertex_indices, vertex2_feature, search_eps, search_k);

          this->hop_sum_ += top_list.hop_sum_;
          this->dist_cal_sum_ += top_list.dist_cal_sum_;
          this->checked_vertices_sum_ += top_list.checked_vertices_sum_;
          this->search_count_++;

          // find a good new vertex3
          float best_gain = total_gain;
          float dist23 = -1;
          float dist34 = -1;

          // We use the descending order to find the worst swap combination with the best gain
          // Sometimes the gain between the two best combinations is the same, its better to use one with the bad edges to make later improvements easier
          for(auto&& result : topListDescending(top_list)) {
            uint32_t new_vertex3 = result.getInternalIndex();

            // vertex1 and vertex2 got tested in the recursive call before and vertex4 got just disconnected from vertex2
            if(vertex1 != new_vertex3 && vertex2 != new_vertex3 && graph.hasEdge(vertex2, new_vertex3) == false) {

              // does the vertex new_vertex3 has a neighbor which is connected to vertex2 and has a lower distance?
              // if(useRNG && steps == 0 && deglib::analysis::checkRNG(graph, edges_per_vertex, new_vertex3, vertex2, result.getDistance()) == false) 
              //   continue;

              // if(deglib::analysis::checkRNG(graph, edges_per_vertex, new_vertex3, vertex2, result.getDistance()) == false) 
              //    continue;

              // 1.1 When vertex2 and the new vertex 3 gets connected full graph connectivity is assured again, 
              //     but the subgraph between vertex1/vertex2 and vertex3/vertex4 might just have one edge(vertex2, vertex3).
              //     Furthermore Node 3 has now to many edges, find an good edge to remove to improve the overall graph distortion. 
              //     FYI: If the just selected vertex3 is the same as the old vertex3, this process might cut its connection to vertex4 again.
              //     This will be fixed in the next step or until the recursion reaches max_path_length.
              const auto neighbor_weights = graph.getNeighborWeights(new_vertex3);
              const auto neighbor_indices = graph.getNeighborIndices(new_vertex3);

              // float avg_weight = 0;
              // float sum2_weight = 0;
              // for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
              //   const auto w = neighbor_weights[edge_idx];
              //   avg_weight += w;
              //   sum2_weight += w*w;
              // }
              // avg_weight /= edges_per_vertex;
              // float avg_variance = std::sqrt(sum2_weight/edges_per_vertex - avg_weight*avg_weight);

              for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
                uint32_t new_vertex4 = neighbor_indices[edge_idx];

                // find a non-RNG edge between the new_vertex3 and new_vertex4, which would be RNG conform between vertex2 and new_vertex4
                // auto factor = 1.0f;
                // if(steps == 0 && deglib::analysis::checkRNG(graph, edges_per_vertex, new_vertex3, new_vertex4, neighbor_weights[edge_idx]) == false) 
                //     factor *= rng_factor_;
                // if(steps == 0 && deglib::analysis::checkRNG(graph, edges_per_vertex, vertex2, new_vertex4, dist_func(vertex2_feature, graph.getFeatureVector(new_vertex4), dist_func_param)))
                //   factor *= rng_factor_/2;

                // compute the gain of the graph distortion if this change would be applied
                const auto factor = 0; //(steps > 0 || deglib::analysis::checkRNG(graph, edges_per_vertex, new_vertex3, new_vertex4, neighbor_weights[edge_idx])) ? 0.0f : avg_variance;
                //const auto factor = (steps > 0 || deglib::analysis::checkRNG(graph, edges_per_vertex, new_vertex3, new_vertex4, neighbor_weights[edge_idx])) ? 1.0f : std::max(1.0f, neighbor_weights[edge_idx]/avg_weight);
                const auto gain = total_gain - result.getDistance() + neighbor_weights[edge_idx] + factor;

                // do not remove the edge which was just added
                if(new_vertex4 != vertex2 && best_gain < gain) {
                  best_gain = gain;
                  vertex3 = new_vertex3;
                  vertex4 = new_vertex4;
                  dist23 = result.getDistance();
                  dist34 = neighbor_weights[edge_idx];    
                }
              }
            }
          }

          // no new vertex3 was found
          if(dist23 == -1)
            return false;

          // replace the temporary self-loop of vertex2 with a connection to vertex3. 
          total_gain = (total_gain - dist23) + dist34;
          graph.changeEdge(vertex2, vertex2, vertex3, dist23);
          changes.emplace_back(vertex2, vertex2, 0.f, vertex3, dist23);

          // 1.2 Remove the worst edge of vertex3 to vertex4 and replace it with the connection to vertex2
          //     Add a temporaty self-loop for vertex4 for the missing edge to vertex3
          graph.changeEdge(vertex3, vertex4, vertex2, dist23);
          changes.emplace_back(vertex3, vertex4, dist34, vertex2, dist23);
          graph.changeEdge(vertex4, vertex3, vertex4, 0.f);
          changes.emplace_back(vertex4, vertex3, dist34, vertex4, 0.f);
        }

        // There is no step 2, since step 1 and 2 got combined in the method above with a different heuristic
      }

      // 3. Try to connect vertex1 with vertex4
      {
        const auto& feature_space = this->graph_.getFeatureSpace();
        const auto dist_func = feature_space.get_dist_func();
        const auto dist_func_param = feature_space.get_dist_func_param();

        // 3.1a Node1 and vertex4 might be the same. This is quite the rare case, but would mean there are two edges missing.
        //     Proceed like extending the graph:
        //     Search for a good vertex to connect to, remove its worst edge and connect
        //     both vertices of the worst edge to the vertex4. Skip the edge any of the two
        //     two vertices are already connected to vertex4.
        if(vertex1 == vertex4) {

          // finds and keeps the best possible connection for vertex 4, 
          // even if other vertices do not get ideal connections with this trade
          if(high_variance_swaps) {
  
            // find a good (not yet connected) vertex for vertex1/vertex4
            const std::vector<uint32_t> entry_vertex_indices = { vertex2, vertex3 };
            const auto vertex4_feature = graph.getFeatureVector(vertex4);
            auto top_list = graph.search(entry_vertex_indices, vertex4_feature, search_eps, search_k);

            this->hop_sum_ += top_list.hop_sum_;
            this->dist_cal_sum_ += top_list.dist_cal_sum_;
            this->checked_vertices_sum_ += top_list.checked_vertices_sum_;
            this->search_count_++;

            for(auto&& result : topListAscending(top_list)) {
              const auto good_vertex = result.getInternalIndex();

              // the new vertex should not be connected to vertex4 yet
              if(vertex4 != good_vertex && graph.hasEdge(vertex4, good_vertex) == false) {
                const auto good_vertex_dist = result.getDistance();

                // select any edge of the good vertex which improves the graph quality when replaced with a connection to vertex 4
                const auto neighbors_indices = graph.getNeighborIndices(good_vertex);
                const auto neighbor_weights = graph.getNeighborWeights(good_vertex);
                for (size_t i = 0; i < edges_per_vertex; i++) {
                  const auto selected_neighbor = neighbors_indices[i];

                  // ignore edges where the second vertex is already connect to vertex4
                  if(vertex4 != selected_neighbor && graph.hasEdge(vertex4, selected_neighbor) == false) {
                    const auto old_neighbor_dist = neighbor_weights[i];
                    const auto new_neighbor_dist = dist_func(vertex4_feature, graph.getFeatureVector(selected_neighbor), dist_func_param);

                    // do all the changes improve the graph?
                    if((total_gain + old_neighbor_dist) - (good_vertex_dist + new_neighbor_dist) > 0) {

                      // replace the two self-loops of vertex4/vertex1 with a connection to the good vertex and its selected neighbor
                      graph.changeEdge(vertex4, vertex4, good_vertex, good_vertex_dist);
                      changes.emplace_back(vertex4, vertex4, 0.f, good_vertex, good_vertex_dist);
                      graph.changeEdge(vertex4, vertex4, selected_neighbor, new_neighbor_dist);
                      changes.emplace_back(vertex4, vertex4, 0.f, selected_neighbor, new_neighbor_dist);

                      // replace from good vertex the connection to the selected neighbor with one to vertex4
                      graph.changeEdge(good_vertex, selected_neighbor, vertex4, good_vertex_dist);
                      changes.emplace_back(good_vertex, selected_neighbor, old_neighbor_dist, vertex4, good_vertex_dist);

                      // replace from the selected neighbor the connection to the good vertex with one to vertex4
                      graph.changeEdge(selected_neighbor, good_vertex, vertex4, new_neighbor_dist);
                      changes.emplace_back(selected_neighbor, good_vertex, old_neighbor_dist, vertex4, new_neighbor_dist);

                      return true;
                    }
                  }
                }
              }
            }
          } 
          else
          {
            // find a good (not yet connected) vertex for vertex1/vertex4
            const std::vector<uint32_t> entry_vertex_indices = { vertex2, vertex3 };
            const auto vertex4_feature = graph.getFeatureVector(vertex4);
            auto top_list = graph.search(entry_vertex_indices, vertex4_feature, search_eps, search_k);

            this->hop_sum_ += top_list.hop_sum_;
            this->dist_cal_sum_ += top_list.dist_cal_sum_;
            this->checked_vertices_sum_ += top_list.checked_vertices_sum_;
            this->search_count_++;

            float best_gain = 0;
            uint32_t best_selected_neighbor = 0;
            float best_old_neighbor_dist = 0;
            float best_new_neighbor_dist = 0;
            uint32_t best_good_vertex = 0;
            float best_good_vertex_dist = 0;
            for(auto&& result : topListAscending(top_list)) {
              const auto good_vertex = result.getInternalIndex();

              // the new vertex should not be connected to vertex4 yet
              if(vertex4 != good_vertex && graph.hasEdge(vertex4, good_vertex) == false) {
                const auto good_vertex_dist = result.getDistance();

              // does the vertex good_vertex has a neighbor which is connected to vertex4 and has a lower distance?
              // if(steps == 0 && deglib::analysis::checkRNG(graph, edges_per_vertex, good_vertex, vertex4, good_vertex_dist) == false) 
              //   continue;

                // select any edge of the good vertex which improves the graph quality when replaced with a connection to vertex 4
                const auto neighbors_indices = graph.getNeighborIndices(good_vertex);
                const auto neighbor_weights = graph.getNeighborWeights(good_vertex);
                for (size_t i = 0; i < edges_per_vertex; i++) {
                  const auto selected_neighbor = neighbors_indices[i];

                  // ignore edges where the second vertex is already connect to vertex4
                  if(vertex4 != selected_neighbor && graph.hasEdge(vertex4, selected_neighbor) == false) {
                    const auto factor = 1;
                    // const auto factor = deglib::analysis::checkRNG(graph, edges_per_vertex, good_vertex, selected_neighbor, neighbor_weights[i]) ? 1.0f : rng_factor_;
                    const auto old_neighbor_dist = neighbor_weights[i];
                    const auto new_neighbor_dist = dist_func(vertex4_feature, graph.getFeatureVector(selected_neighbor), dist_func_param);


                    // do all the changes improve the graph?
                    float new_gain = (total_gain + old_neighbor_dist) - (good_vertex_dist + new_neighbor_dist);
                    if(best_gain < new_gain) {
                      best_gain = new_gain;
                      best_selected_neighbor = selected_neighbor;
                      best_old_neighbor_dist = old_neighbor_dist;
                      best_new_neighbor_dist = new_neighbor_dist;
                      best_good_vertex = good_vertex;
                      best_good_vertex_dist = good_vertex_dist;
                    }
                  }
                }
              }
            }

            if(best_gain > 0)
            {

              // replace the two self-loops of vertex4/vertex1 with a connection to the good vertex and its selected neighbor
              graph.changeEdge(vertex4, vertex4, best_good_vertex, best_good_vertex_dist);
              changes.emplace_back(vertex4, vertex4, 0.f, best_good_vertex, best_good_vertex_dist);
              graph.changeEdge(vertex4, vertex4, best_selected_neighbor, best_new_neighbor_dist);
              changes.emplace_back(vertex4, vertex4, 0.f, best_selected_neighbor, best_new_neighbor_dist);

              // replace from good vertex the connection to the selected neighbor with one to vertex4
              graph.changeEdge(best_good_vertex, best_selected_neighbor, vertex4, best_good_vertex_dist);
              changes.emplace_back(best_good_vertex, best_selected_neighbor, best_old_neighbor_dist, vertex4, best_good_vertex_dist);

              // replace from the selected neighbor the connection to the good vertex with one to vertex4
              graph.changeEdge(best_selected_neighbor, best_good_vertex, vertex4, best_new_neighbor_dist);
              changes.emplace_back(best_selected_neighbor, best_good_vertex, best_old_neighbor_dist, vertex4, best_new_neighbor_dist);

              return true;
            }
          }


        } else {

          // 3.1b If there is a way from vertex2 or vertex3, to vertex1 or vertex4 then ...
				  //      Try to connect vertex1 with vertex4
          //      Much more likly than 3.1a 
				  if(graph.hasEdge(vertex1, vertex4) == false) {

            // Is the total of all changes still beneficial?
            const auto dist14 = dist_func(graph.getFeatureVector(vertex1), graph.getFeatureVector(vertex4), dist_func_param);
            if((total_gain - dist14) > 0) {

              const std::vector<uint32_t> entry_vertex_indices = { vertex2, vertex3 }; 
              if(graph.hasPath(entry_vertex_indices, vertex1, this->improve_eps_, this->improve_k_).size() > 0 || graph.hasPath(entry_vertex_indices, vertex4, this->improve_eps_, improve_k_).size() > 0) {
                
                // replace the the self-loops of vertex1 with a connection to the vertex4
                graph.changeEdge(vertex1, vertex1, vertex4, dist14);
                changes.emplace_back(vertex1, vertex1, 0.f, vertex4, dist14);

                // replace the the self-loops of vertex4 with a connection to the vertex1
                graph.changeEdge(vertex4, vertex4, vertex1, dist14);
                changes.emplace_back(vertex4, vertex4, 0.f, vertex1, dist14);

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
      
      // 5. swap vertex1 and vertex4 every second round, to give each a fair chance
      if(steps % 2 == 1) {
        uint32_t b = vertex1;
        vertex1 = vertex4;
        vertex4 = b;
      }

      // 6. early stop
      // TODO Since an edge is missing the total gain should always be higher zero. A better heuristic would be total_gain-<current worst edge of vertex4>.
      // In the next iteration we try to find a good edge for vertex 4 which is probably better than the current worst edge of vertex 4.
      // But since the total gain could already be so bad that even if the new edge is better, the overall gain would still be negative, we should stop here.
      if(total_gain < 0) {
        //fmt::print("Current swap path is degenerating the graph. Rollback.\n");
        return false;
      }

      return improveEdges(changes, vertex1, vertex4, vertex2, vertex3, total_gain, steps + 1);
    }

    bool improveEdges() {

      auto& graph = this->graph_;
      const auto edges_per_vertex = graph.getEdgesPerNode();

      // 1. remove the worst edge of a random vertex 

      // 1.1 select a random vertex
      auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
      uint32_t vertex1 = distrib(this->rnd_);

      // auto boolean_distrib = std::uniform_int_distribution<uint32_t>(0, 1);
      bool find_rng = false; //boolean_distrib(this->rnd_) == 1;

      // 1.2 find the worst edge of this vertex
      uint32_t bad_neighbor_index = 0;
      float bad_neighbor_weight = -1.0f;
      bool bad_is_rng = true;
      const auto neighbor_weights = graph.getNeighborWeights(vertex1);
      const auto neighbor_indices = graph.getNeighborIndices(vertex1);
      for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
        if(find_rng) {
          const auto is_rng = deglib::analysis::checkRNG(graph, edges_per_vertex, vertex1, neighbor_indices[edge_idx], neighbor_weights[edge_idx]);
       
          if((bad_is_rng == true && is_rng == false) || (bad_neighbor_weight < neighbor_weights[edge_idx] && bad_is_rng == is_rng)) {
            bad_neighbor_index = neighbor_indices[edge_idx];
            bad_neighbor_weight = neighbor_weights[edge_idx];   
            bad_is_rng = is_rng;
          }
        } else if(bad_neighbor_weight < neighbor_weights[edge_idx]) {
            bad_neighbor_index = neighbor_indices[edge_idx];
            bad_neighbor_weight = neighbor_weights[edge_idx];
        }  
      }

      // nothing found
      if(bad_neighbor_weight < 0)
        return false;

      return improveEdges(vertex1, bad_neighbor_index, bad_neighbor_weight);
    }

    bool improveEdges(uint32_t vertex1, uint32_t vertex2, float dist12) {
      auto changes = std::vector<deglib::builder::BuilderChange>();

      // remove the edge between vertex 1 and vertex 2 (add temporary self-loops)
      auto& graph = this->graph_;
      graph.changeEdge(vertex1, vertex2, vertex1, 0.f);
      changes.emplace_back(vertex1, vertex2, dist12, vertex1, 0.f);
      graph.changeEdge(vertex2, vertex1, vertex2, 0.f);
      changes.emplace_back(vertex2, vertex1, dist12, vertex2, 0.f);

      if(improveEdges(changes, vertex1, vertex2, vertex1, vertex1, dist12, 0) == false) {

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
      const auto edge_per_vertex = this->graph_.getEdgesPerNode();

      // // try to build an initial graph, containing the minium amount of vertices (edge_per_vertex + 1)
      // const auto edge_per_vertex_p1 = (uint8_t)(edge_per_vertex + 1);
      // if(graph_.size() < edge_per_vertex_p1) {

      //   // graph should be empty to initialize
      //   if(this->graph_.size() > 0) {
      //     fmt::print(stderr, "graph has already {} vertices and can therefore not be initialized \n", this->graph_.size());
      //     perror("");
      //     abort();
      //   }

      //   // wait until enough new entries exists to build the initial graph
      //   while(new_entry_queue_.size() < edge_per_vertex_p1)
      //     std::this_thread::sleep_for(std::chrono::milliseconds(1000));

      //   // setup the initial graph
      //   {
      //     std::array<BuilderAddTask, std::numeric_limits<uint8_t>::max()> initial_entries;
      //     std::copy(new_entry_queue_.begin(), std::next(new_entry_queue_.begin(), edge_per_vertex_p1), initial_entries.begin());
      //     new_entry_queue_.erase(new_entry_queue_.begin(), std::next(new_entry_queue_.begin(), edge_per_vertex_p1));
      //     initialGraph({initial_entries.data(), edge_per_vertex_p1});
      //   }

      //   // inform the callback about the initial graph
      //   status.added += edge_per_vertex_p1;
      //   callback(status);
      // } 
      // else 
      // {
      //   status.added = graph_.size();
      // }

      // run a loop to add, delete and improve the graph
      do{

        // add or delete a vertex
        if(this->new_entry_queue_.size() > 0 || this->remove_entry_queue_.size() > 0) {
          auto add_task_manipulation_index = std::numeric_limits<uint64_t>::max();
          auto del_task_manipulation_index = std::numeric_limits<uint64_t>::max();

          if(this->new_entry_queue_.size() > 0) 
            add_task_manipulation_index = this->new_entry_queue_.front().manipulation_index;

          if(this->remove_entry_queue_.size() > 0) 
            del_task_manipulation_index = this->remove_entry_queue_.front().manipulation_index;

          if(add_task_manipulation_index < del_task_manipulation_index) {
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
        if(graph_.size() > edge_per_vertex && improve_k_ > 0) {
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