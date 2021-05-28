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
  uint64_t timestamp;
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
      const auto edges_per_node = graph.getEdgesPerNode();
      for (auto &&entry : entries)   
        graph.addNode(entry.label, entry.feature.data());
      
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
      const auto edges_per_node = graph.getEdgesPerNode();
      const auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
      const std::vector<uint32_t> entry_node_indizies = { distrib(this->rnd_) };
      auto result_queue = graph.yahooSearch(entry_node_indizies, add_task.feature.data(), this->extend_eps_, std::max(this->extend_k_, edges_per_node));

      // their should always be enough neighbors (search results), otherwise the graph would be broken
      if(result_queue.size() < edges_per_node) {
        fmt::print(stderr, "the graph search for the new node {} did only provided {} results \n", external_label, result_queue.size());
        perror("");
        abort();
      }

      // add an empty node to the graph (no neighbor information yet)
      const auto internal_index = graph.addNode(external_label, add_task.feature.data());

      // for computing distances to neighbors not in the result queue
      const auto dist_func = graph.getFeatureSpace().get_dist_func();
      const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
        
      // remove the worst edge of the good neighbors and connect them with this new node
      auto new_neighbors = std::vector<std::pair<uint32_t, float>>();
      while(new_neighbors.size() < edges_per_node) {
        const auto& result = result_queue.front();

        // check if the node is already in the edge list of the new node (added during a previous loop-run)
        // since all edges are undirected and the edge information of the new node does not yet exist, we search the other way around.
        if(graph.hasEdge(result.getInternalIndex(), internal_index)) {
          result_queue.pop();
          continue;
        }

        // find the worst edge of the new neighbor
        uint32_t bad_neighbor_index = 0;
        float bad_neighbor_weight = 0;
        const auto neighbor_weights = graph.getNeighborWeights(result.getInternalIndex());
        const auto neighbor_indizies = graph.getNeighborIndizies(result.getInternalIndex());
        for (size_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {
          const auto neighbor_index = neighbor_indizies[edge_idx];
          const auto neighbor_weight = neighbor_weights[edge_idx];

          // the new node might have been added to the neighbor list of this result-node during a previous loop
          // the suggest neighbors might already be in the edge list of the new node
          // the weight of the neighbor might not be worst than the current worst one
          if(internal_index != neighbor_index && graph.hasEdge(neighbor_index, internal_index) == false && bad_neighbor_weight < neighbor_weight) {
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
        graph.changeEdge(result.getInternalIndex(), bad_neighbor_index, internal_index, result.getDistance());
        new_neighbors.emplace_back(result.getInternalIndex(), result.getDistance());

        // place the new node in the edge list of the result-node and its worst edge neighbor
        const auto distance = dist_func(add_task.feature.data(), graph.getFeatureVector(bad_neighbor_index), dist_func_param);
        graph.changeEdge(bad_neighbor_index, result.getInternalIndex(), internal_index, distance);
        new_neighbors.emplace_back(bad_neighbor_index, distance);

        // iterate to the next search result
        result_queue.pop();
      }

      // sort the neighbors by their neighbor indizies and store them in the new node
      std::sort(new_neighbors.begin(), new_neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
      auto neighbor_indizies = std::vector<uint32_t>();
      auto neighbor_weights = std::vector<float>();
      for (auto &&neighbor : new_neighbors) {
        neighbor_indizies.emplace_back(neighbor.first);
        neighbor_weights.emplace_back(neighbor.second);
      }
      graph.changeEdges(internal_index, neighbor_indizies.data(), neighbor_weights.data());
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
     * Call just improve() to improve the graph.
     * This method takes an array where all graph changes will be documented.
     * 
     * Runs a single graph improvement step by swapping a series of edges.
     * If those changes improve the graph this method returns true otherwise false. 
     * 
     * @return true if a good sequences of changes has been found
     */
    bool improve(std::vector<deglib::builder::BuilderChange>& changes) {
      auto& graph = this->graph_;
      const auto edges_per_node = graph.getEdgesPerNode();

      // how much does the graph improve with those changes
      float total_gain = 0;

      // 1. remove the worst edge of a random node 
      uint32_t node1, node2;
      float dist12;
      {
        // 1.1 select a random node
        const auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
        node1 = distrib(this->rnd_);

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

        // 1.3 remove the edge between node 1 and node 2 (add temporary self-loops)
        node2 = bad_neighbor_index;
        dist12 = bad_neighbor_weight;
        total_gain = dist12;
        graph.changeEdge(node1, node2, node1, 0.f);
        changes.emplace_back(node1, node2, dist12, node1, 0.f);
        graph.changeEdge(node2, node1, node2, 0.f);
        changes.emplace_back(node2, node1, dist12, node2, 0.f);
      }

      // 2. Find a replacement edge for node2. This edge should connect the potential subgraph of node2
      //    with the potential subgraph of node1. Consider only nodes of the approximate nearest neighbor
      //    search. If the search starts from node1 all nodes in the result list are in the subgraph 
      //    of node1 and would therefore connect the two potential subgraphs of node1 and node2.	
      uint32_t node3 = 0;
      float dist23 = 0;
      {
        // find a good node3 to connect to node 2
        const std::vector<uint32_t> entry_node_indizies = { node1 };
        const auto node2_feature = graph.getFeatureVector(node2);
        auto result_queue = graph.yahooSearch(entry_node_indizies, node2_feature, this->improve_eps_, std::max(this->improve_k_, edges_per_node));
        while(result_queue.size() > 0) {
          const auto& result = result_queue.front();

          if(node1 != result.getInternalIndex() && node2 != result.getInternalIndex() && graph.hasEdge(node2, result.getInternalIndex()) == false) {
            node3 = result.getInternalIndex();
            dist23 = result.getDistance();

            // replace the temporary self-loop of node2 with a connection to node3. 
            graph.changeEdge(node2, node2, node3, dist23);
            changes.emplace_back(node2, node2, 0.f, node3, dist23);
            total_gain -= dist23;
            break;
          }

          result_queue.pop();
        }

        // no good node3 was found, stop this swap try
        if(dist23 == 0) 
          return false;
      }

      // 3. Node 3 has now to many edges, remove the worst one. Ignore the just added edge. 
		  uint32_t node4;
      float dist34;
		  {
        // 3.1 find the worst edge of node3
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

        // 3.2 Remove the worst edge of node3 to node4 and replace it with the connection to node2
        //     Add a temporaty self-loop for node4 for the missing edge to node3
        node4 = bad_neighbor_index;
        dist34 = bad_neighbor_weight;
        total_gain += dist34;
        graph.changeEdge(node3, node4, node2, dist23);
        changes.emplace_back(node3, node4, dist34, node2, dist23);
        graph.changeEdge(node4, node3, node4, 0.f);
        changes.emplace_back(node4, node3, dist34, node4, 0.f);
      }

      // 4. Try to connect node1 with node4
      {
        const auto& feature_space = this->graph_.getFeatureSpace();
        const auto dist_func = feature_space.get_dist_func();
        const auto dist_func_param = feature_space.get_dist_func_param();

        // 4.1a Node1 and node4 might be the same. Proceed like extending the graph.
        //     Search for a good node to connect to, remove its worst edge and connect
        //     both nodes of the worst edge to the node4. Skip the edge any of the two
        //     two nodes are already connected to node4.
        if(node1 == node4) {

          // find a good (not yet connected) node for node1/node4
          const std::vector<uint32_t> entry_node_indizies = { node2, node3 };
          const auto node4_feature = graph.getFeatureVector(node4);
          auto result_queue = graph.yahooSearch(entry_node_indizies, node4_feature, this->improve_eps_, std::max(this->improve_k_, edges_per_node));

          while(result_queue.size() > 0) {
            const auto& result = result_queue.front();
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

            result_queue.pop();
          }
        }

        // TODO when improve_recursive is implemented it might be better to remove this section
        else {

          // If there is a way from node2 or node3, to node1 or node4 then ...
				  // 4.1b Try to connect node1 with node4
				  if(graph.hasEdge(node1, node4) == false) {

            // Is the total of all changes still beneficial?
            const auto dist14 = dist_func(graph.getFeatureVector(node1), graph.getFeatureVector(node4), dist_func_param);
            if((total_gain - dist14) > 0) {

              const std::vector<uint32_t> entry_node_indizies = { node2, node3 }; 
              if(graph.hasPath(entry_node_indizies, node1, improve_eps_, improve_k_).size() > 0 || graph.hasPath(entry_node_indizies, node4, improve_eps_, improve_k_).size() > 0) {
                
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


      return false;
    }

    /**
     * Runs a single graph improvement step. A sequence of edges will be swapped.
     * If the overall graph quality improve those changes will be keep overwise reverted.
     */
    bool improve() {
      auto changes = std::vector<deglib::builder::BuilderChange>();

      if(improve(changes) == false) {

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
          const auto& entry = this->new_entry_queue_.front();
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
          auto add_task_timestamp = std::numeric_limits<uint64_t>::max();
          auto del_task_timestamp = std::numeric_limits<uint64_t>::max();

          if(this->new_entry_queue_.size() > 0) 
            add_task_timestamp = this->new_entry_queue_.front().timestamp;

          if(this->remove_entry_queue_.size() > 0) 
            del_task_timestamp = this->remove_entry_queue_.front().timestamp;

          if(add_task_timestamp < del_task_timestamp) {
            extendGraph(this->new_entry_queue_.front());
            status.added++;
            this->new_entry_queue_.pop();
          } else {
            shrinkGraph(this->remove_entry_queue_.front());
            status.deleted++;
            this->remove_entry_queue_.pop();
          }
        }

        // try to improve the graph
        for (int swap_try = 0; swap_try < int(this->swap_tries_); swap_try++) {
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