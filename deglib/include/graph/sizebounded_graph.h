#pragma once

#include <cstdint>
#include <limits>
#include <queue>
#include <math.h>

#include <fmt/core.h>
#include <tsl/robin_hash.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "graph.h"
#include "repository.h"
#include "search.h"

namespace deglib::graph
{

/**
 * A size bounded undirected and weighted n-regular graph.
 * 
 * The node count and number of edges per nodes is bounded to a fixed value at 
 * construction time. The graph is therefore n-regular where n is the number of 
 * eddes per node.
 * 
 * Furthermode the graph is undirected, if there is connection from A to B than 
 * there musst be one from B to A. All connections are stored in the neighbor 
 * indizies list of every node. The indizies are based on the indizies of their 
 * corresponding nodes. Each node has an index and an external label. The index 
 * is for internal computation and goes from 0 to the number of nodes. Where 
 * the external label can be any signed 32-bit integer. The indizies in the 
 * neighbors list are ascending sorted.
 * 
 * Every edge contains of a neighbor node index and a weight. The weights and
 * neighbor indizies are in separated list, but have the same order.
 * 
 * The number of nodes is limited to uint32.max
 */
class SizeBoundedGraph : public deglib::graph::MutableGraph {

  using SEARCHFUNC = deglib::search::ResultSet (*)(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k);

  inline static deglib::search::ResultSet searchL2Ext16(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k) {
    return graph.yahooSearchImpl<deglib::distances::L2Float16Ext>(entry_node_indizies, query, eps, k);
  }

  inline static deglib::search::ResultSet searchL2Ext4(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k) {
    return graph.yahooSearchImpl<deglib::distances::L2Float4Ext>(entry_node_indizies, query, eps, k);
  }

  inline static deglib::search::ResultSet searchL2Ext16Residual(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k) {
    return graph.yahooSearchImpl<deglib::distances::L2Float16ExtResiduals>(entry_node_indizies, query, eps, k);
  }

  inline static deglib::search::ResultSet searchL2Ext4Residual(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k) {
    return graph.yahooSearchImpl<deglib::distances::L2Float4ExtResiduals>(entry_node_indizies, query, eps, k);
  }

  static SEARCHFUNC getSearchFunction(const size_t dim) {
    if (dim % 16 == 0)
      return deglib::graph::SizeBoundedGraph::searchL2Ext16;
    else if (dim % 4 == 0)
      return deglib::graph::SizeBoundedGraph::searchL2Ext4;
    else if (dim > 16)
      return deglib::graph::SizeBoundedGraph::searchL2Ext16Residual;
    else if (dim > 4)
      return deglib::graph::SizeBoundedGraph::searchL2Ext4Residual;
    else
      return deglib::graph::SizeBoundedGraph::searchL2Ext16;
  }

  static uint32_t compute_aligned_byte_size_per_node(const uint8_t edges_per_node, const uint16_t feature_byte_size, const uint8_t alignment) {
    const uint32_t byte_size = uint32_t(feature_byte_size) + uint32_t(edges_per_node) * (sizeof(uint32_t) + sizeof(float)) + sizeof(uint32_t);
    if (alignment == 0)
      return byte_size;
    else {
      return ((byte_size + alignment - 1) / alignment) * alignment;
    }
  }

  static std::byte* compute_aligned_pointer(const std::unique_ptr<std::byte[]>& arr, const uint8_t alignment) {
    if (alignment == 0)
      return arr.get();
    else {
      void* ptr = arr.get();
      size_t space = std::numeric_limits<size_t>::max();
      std::align(alignment, 0, ptr, space);
      return static_cast<std::byte*>(ptr);
    }
  }

  // alignment of node information in bytes (all feature vectors will be 256bit aligned for faster SIMD processing)
  static const uint8_t object_alignment = 32; 

  const uint32_t max_node_count_;
  const uint8_t edges_per_node_;
  const uint16_t feature_byte_size_;

  const uint32_t byte_size_per_node_;
  const uint32_t neighbor_indizies_offset_;
  const uint32_t neighbor_weights_offset_;
  const uint32_t external_label_offset_;

  // list of nodes (node: std::byte* feature vector, uint32_t* indizies of neighbor nodes, float* weights of neighbor nodes, uint32_t external label)      
  std::unique_ptr<std::byte[]> nodes_;
  std::byte* nodes_memory_;

  // map from the label of a node to the internal node index
  tsl::robin_map<uint32_t, uint32_t> label_to_index_;

  // internal search function with embedded distances function
  const SEARCHFUNC search_func_;

    // distance calculation function between feature vectors of two graph nodes
  const deglib::L2Space feature_space_;

 public:
  SizeBoundedGraph(const uint32_t max_node_count, const uint8_t edges_per_node, const deglib::L2Space feature_space)
      : edges_per_node_(edges_per_node), 
        max_node_count_(max_node_count), 
        feature_space_(feature_space),
        search_func_(getSearchFunction(feature_space.dim())),
        feature_byte_size_(uint16_t(feature_space.get_data_size())), 
        byte_size_per_node_(compute_aligned_byte_size_per_node(edges_per_node, uint16_t(feature_space.get_data_size()), object_alignment)), 
        neighbor_indizies_offset_(uint32_t(feature_space.get_data_size())),
        neighbor_weights_offset_(neighbor_indizies_offset_ + uint32_t(edges_per_node) * sizeof(uint32_t)),
        external_label_offset_(neighbor_weights_offset_ + uint32_t(edges_per_node) * sizeof(float)), 
        nodes_(std::make_unique<std::byte[]>(size_t(max_node_count) * byte_size_per_node_ + object_alignment)), 
        nodes_memory_(compute_aligned_pointer(nodes_, object_alignment)), 
        label_to_index_(max_node_count) {
  }

  /**
   *  Load from file
   */
  SizeBoundedGraph(const uint32_t max_node_count, const uint8_t edges_per_node, const deglib::L2Space feature_space, std::ifstream& ifstream, const uint32_t size)
      : SizeBoundedGraph(max_node_count, edges_per_node, feature_space) {

    // copy the old data over
    uint32_t file_byte_size_per_node = compute_aligned_byte_size_per_node(this->edges_per_node_, this->feature_byte_size_, 0);
    for (uint32_t i = 0; i < size; i++) {
      ifstream.read(reinterpret_cast<char*>(this->node_by_index(i)), file_byte_size_per_node);
      label_to_index_.emplace(this->getExternalLabel(i), i);
    }
  }

  /**
   * Current maximal capacity of nodes
   */ 
  const auto capacity() const {
    return this->max_node_count_;
  }

  /**
   * Number of nodes in the graph
   */
  const uint32_t size() const override {
    return (uint32_t) this->label_to_index_.size();
  }

  /**
   * Number of edges per node 
   */
  const uint8_t getEdgesPerNode() const override {
    return this->edges_per_node_;
  }

  const deglib::SpaceInterface<float>& getFeatureSpace() const override {
    return this->feature_space_;
  }

private:  
  inline std::byte* node_by_index(const uint32_t internal_idx) const {
    return nodes_memory_ + internal_idx * byte_size_per_node_;
  }

  inline const uint32_t label_by_index(const uint32_t internal_idx) const {
    return *reinterpret_cast<const int32_t*>(node_by_index(internal_idx) + external_label_offset_);
  }

  inline const std::byte* feature_by_index(const uint32_t internal_idx) const{
    return node_by_index(internal_idx);
  }

  inline const uint32_t* neighbors_by_index(const uint32_t internal_idx) const {
    return reinterpret_cast<uint32_t*>(node_by_index(internal_idx) + neighbor_indizies_offset_);
  }

  inline const float* weights_by_index(const uint32_t internal_idx) const {
    return reinterpret_cast<const float*>(node_by_index(internal_idx) + neighbor_weights_offset_);
  }

public:

  /**
   * convert an external label to an internal index
   */ 
  inline const uint32_t getInternalIndex(const uint32_t external_label) const override {
    return label_to_index_.find(external_label)->second;
  }

  inline const uint32_t getExternalLabel(const uint32_t internal_idx) const override {
    return label_by_index(internal_idx);
  }

  inline const std::byte* getFeatureVector(const uint32_t internal_idx) const override{
    return feature_by_index(internal_idx);
  }

  inline const uint32_t* getNeighborIndizies(const uint32_t internal_idx) const override {
    return neighbors_by_index(internal_idx);
  }

  inline const float* getNeighborWeights(const uint32_t internal_idx) const override {
    return weights_by_index(internal_idx);
  }

  inline const bool hasNode(const uint32_t external_label) const override {
    return label_to_index_.contains(external_label);
  }

  inline const bool hasEdge(const uint32_t internal_index, const uint32_t neighbor_index) const override {
    auto neighbor_indizies = getNeighborIndizies(internal_index);
    auto neighbor_indizies_end = neighbor_indizies + this->edges_per_node_;  
    auto neighbor_ptr = std::lower_bound(neighbor_indizies, neighbor_indizies_end, neighbor_index); 
    return (*neighbor_ptr == neighbor_index);
  }

  const bool saveGraph(const char* path_to_graph) const override {
    auto out = std::ofstream(path_to_graph, std::ios::out | std::ios::binary);

    // check open file for write
    if (!out.is_open()) {
      fmt::print(stderr, "Error in open file {}\n", path_to_graph);
      return false;
    }

    // store feature space information
    uint8_t data_type = 1; // 1=float
    out.write(reinterpret_cast<const char*>(&data_type), sizeof(data_type));
    uint16_t dim = uint16_t(this->feature_space_.dim());
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

    // store graph information
    uint32_t size = uint32_t(this->size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(&this->edges_per_node_), sizeof(this->edges_per_node_));

    // store the existing nodes
    uint32_t byte_size_per_node = compute_aligned_byte_size_per_node(this->edges_per_node_, this->feature_byte_size_, 0);
    for (uint32_t i = 0; i < size; i++)
      out.write(reinterpret_cast<const char*>(this->node_by_index(i)), byte_size_per_node);    
    out.close();

    return true;
  }

  /**
   * Add a new node. The neighbor indizies will be prefilled with a self-loop, the weights will be 0.
   * 
   * @return the internal index of the new node
   */
  uint32_t addNode(const uint32_t external_label, const std::byte* feature_vector) override {
    const auto new_internal_index = static_cast<uint32_t>(label_to_index_.size());
    label_to_index_.emplace(external_label, new_internal_index);

    auto node_memory = node_by_index(new_internal_index);
    std::memcpy(node_memory, feature_vector, feature_byte_size_);
    std::fill_n(reinterpret_cast<uint32_t*>(node_memory + neighbor_indizies_offset_), this->edges_per_node_, new_internal_index); // temporary self loop
    std::fill_n(reinterpret_cast<float*>(node_memory + neighbor_weights_offset_), this->edges_per_node_, float(0)); // 0 weight
    std::memcpy(node_memory + external_label_offset_, &external_label, sizeof(uint32_t));

    return new_internal_index;
  }

  /**
   * Swap a neighbor with another neighbor and its weight.
   * 
   * @param internal_index node index which neighbors should be changed
   * @param from_neighbor_index neighbor index to remove
   * @param to_neighbor_index neighbor index to add
   * @param to_neighbor_weight weight of the neighbor to add
   * @return true if the from_neighbor_index was found and changed
   */
  bool changeEdge(const uint32_t internal_index, const uint32_t from_neighbor_index, const uint32_t to_neighbor_index, const float to_neighbor_weight) override {
    auto node_memory = node_by_index(internal_index);

    auto neighbor_indizies = reinterpret_cast<uint32_t*>(node_memory + neighbor_indizies_offset_);    // list of neighbor indizizes
    auto neighbor_indizies_end = neighbor_indizies + this->edges_per_node_;                           // end of the list
    auto from_ptr = std::lower_bound(neighbor_indizies, neighbor_indizies_end, from_neighbor_index);  // possible position of the from_neighbor_index in the neighbor list

    // from_neighbor_index not found in the neighbor list
    if(*from_ptr != from_neighbor_index)
      return false;

    auto neighbor_weights = reinterpret_cast<float*>(node_memory + neighbor_weights_offset_);         // list of neighbor weights
    auto to_ptr = std::lower_bound(neighbor_indizies, neighbor_indizies_end, to_neighbor_index);      // neighbor in the list which has a lower index number than to_neighbor_index
    auto from_list_idx = uint32_t(from_ptr - neighbor_indizies);                                      // index of the from_neighbor_index in the neighbor list
    auto to_list_idx = uint32_t(to_ptr - neighbor_indizies);                                          // index where to place the to_neighbor_index 

    // Make same space before inserting the new values
    if(from_list_idx < to_list_idx) {
      std::memmove(neighbor_indizies + from_list_idx, neighbor_indizies + from_list_idx + 1, (to_list_idx - from_list_idx) * sizeof(uint32_t)); 
      std::memmove(neighbor_weights + from_list_idx, neighbor_weights + from_list_idx + 1, (to_list_idx - from_list_idx) * sizeof(float)); 
      to_list_idx--;
    } else if(to_list_idx < from_list_idx) {
      std::memmove(neighbor_indizies + to_list_idx + 1, neighbor_indizies + to_list_idx, (from_list_idx - to_list_idx) * sizeof(uint32_t));
      std::memmove(neighbor_weights + to_list_idx + 1, neighbor_weights + to_list_idx, (from_list_idx - to_list_idx) * sizeof(float));
    }

    neighbor_indizies[to_list_idx] = to_neighbor_index;
    neighbor_weights[to_list_idx] = to_neighbor_weight;

    return true;
  }

  /**
   * Change all edges of a node.
   * The neighbor indizies/weights and feature vectors will be copied.
   * The neighbor array need to have enough neighbors to match the edge-per-node count of the graph.
   * The indizies in the neighbor_indizies array must be sorted.
   */
  void changeEdges(const uint32_t internal_index, const uint32_t* neighbor_indizies, const float* neighbor_weights) override {
    auto node_memory = node_by_index(internal_index);
    std::memcpy(node_memory + neighbor_indizies_offset_, neighbor_indizies, uint32_t(edges_per_node_) * sizeof(uint32_t));
    std::memcpy(node_memory + neighbor_weights_offset_, neighbor_weights, uint32_t(edges_per_node_) * sizeof(float));
  }

  /**
   * Performan a yahooSearch but stops when the to_node was found.
   */
  std::vector<deglib::search::ObjectDistance> hasPath(const std::vector<uint32_t>& entry_node_indizies, const uint32_t to_node, const float eps, const uint32_t k) const override
  {
    const auto query = this->feature_by_index(to_node);
    const auto dist_func = this->feature_space_.get_dist_func();
    const auto dist_func_param = this->feature_space_.get_dist_func_param();

    // set of checked node ids
    auto checked_ids = std::vector<bool>(this->size());

    // items to traverse next
    auto next_nodes = deglib::search::UncheckedSet();

    // trackable information 
    auto trackback = tsl::robin_map<uint32_t, deglib::search::ObjectDistance>();

    // result set
    auto results = deglib::search::ResultSet();   

    // copy the initial entry nodes and their distances to the query into the three containers
    for (auto&& index : entry_node_indizies) {
      checked_ids[index] = true;

      const auto feature = reinterpret_cast<const float*>(this->feature_by_index(index));
      const auto distance = dist_func(query, feature, dist_func_param);
      results.emplace(index, distance);
      next_nodes.emplace(index, distance);
      trackback.insert({index, deglib::search::ObjectDistance(index, distance)});
    }

    // search radius
    auto r = std::numeric_limits<float>::max();

    // iterate as long as good elements are in the next_nodes queue     
    auto good_neighbors = std::array<uint32_t, 256>();
    while (next_nodes.empty() == false)
    {
      // next node to check
      const auto next_node = next_nodes.top();
      next_nodes.pop();

      // max distance reached
      if (next_node.getDistance() > r * (1 + eps)) 
        break;

      size_t good_neighbor_count = 0;
      const auto neighbor_indizies = this->neighbors_by_index(next_node.getInternalIndex());
      for (size_t i = 0; i < this->edges_per_node_; i++) {
        const auto neighbor_index = neighbor_indizies[i];

        // found our target node, create a path back to the entry node
        if(neighbor_index == to_node) {
          auto path = std::vector<deglib::search::ObjectDistance>();
          path.emplace_back(to_node, 0.f);
          path.emplace_back(next_node.getInternalIndex(), next_node.getDistance());

          auto last_node = trackback.find(next_node.getInternalIndex());
          while(last_node != trackback.cend() && last_node->first != last_node->second.getInternalIndex()) {
            path.emplace_back(last_node->second.getInternalIndex(), last_node->second.getDistance());
            last_node = trackback.find(last_node->second.getInternalIndex());
          }

          return path;
        }

        // collect 
        if (checked_ids[neighbor_index] == false)  {
          checked_ids[neighbor_index] = true;
          good_neighbors[good_neighbor_count++] = neighbor_index;
        }
      }

      if (good_neighbor_count == 0)
        continue;

      MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[0])));
      for (size_t i = 0; i < good_neighbor_count; i++) {
        MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[std::min(i + 1, good_neighbor_count - 1)])));

        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_feature_vector = this->feature_by_index(neighbor_index);
        const auto neighbor_distance = dist_func(query, neighbor_feature_vector, dist_func_param);
             
        // check the neighborhood of this node later, if its good enough
        if (neighbor_distance <= r * (1 + eps)) {
          next_nodes.emplace(neighbor_index, neighbor_distance);
          trackback.insert({neighbor_index, deglib::search::ObjectDistance(next_node.getInternalIndex(), next_node.getDistance())});

          // remember the node, if its better than the worst in the result list
          if (neighbor_distance < r) {
            results.emplace(neighbor_index, neighbor_distance);

            // update the search radius
            if (results.size() > k) {
              results.pop();
              r = results.top().getDistance();
            }
          }
        }
      }
    }

    // there is no path
    return std::vector<deglib::search::ObjectDistance>();
  }

  /**
   * The result set contains internal indizies. 
   */
  deglib::search::ResultSet yahooSearch(const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k) const override
  {
    return search_func_(*this, entry_node_indizies, query, eps, k);
  }

  /**
   * The result set contains internal indizies. 
   */
  template <typename COMPARATOR>
  deglib::search::ResultSet yahooSearchImpl(const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k) const
  {
    const auto dist_func_param = this->feature_space_.get_dist_func_param();

    // set of checked node ids
    auto checked_ids = std::vector<bool>(this->size());

    // items to traverse next
    auto next_nodes = deglib::search::UncheckedSet();

    // result set
    // TODO: custom priority queue with an internal Variable Length Array wrapped in a macro with linear-scan search and memcopy 
    auto results = deglib::search::ResultSet();   

    // copy the initial entry nodes and their distances to the query into the three containers
    for (auto&& index : entry_node_indizies) {
      checked_ids[index] = true;
      const auto feature = reinterpret_cast<const float*>(this->feature_by_index(index));
      const auto distance = COMPARATOR::compare(query, feature, dist_func_param);
      next_nodes.emplace(index, distance);
      results.emplace(index, distance);
    }

    // search radius
    auto r = std::numeric_limits<float>::max();

    // iterate as long as good elements are in the next_nodes queue     
    auto good_neighbors = std::array<uint32_t, 256>();    // this limits the neighbor count to 256 using Variable Length Array wrapped in a macro
    while (next_nodes.empty() == false)
    {
      // next node to check
      const auto next_node = next_nodes.top();
      next_nodes.pop();

      // max distance reached
      if (next_node.getDistance() > r * (1 + eps)) 
        break;

      size_t good_neighbor_count = 0;
      const auto neighbor_indizies = this->neighbors_by_index(next_node.getInternalIndex());
      for (size_t i = 0; i < this->edges_per_node_; i++) {
        const auto neighbor_index = neighbor_indizies[i];
        if (checked_ids[neighbor_index] == false)  {
          checked_ids[neighbor_index] = true;
          good_neighbors[good_neighbor_count++] = neighbor_index;
        }
      }

      if (good_neighbor_count == 0)
        continue;

      MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[0])));
      for (size_t i = 0; i < good_neighbor_count; i++) {
        MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[std::min(i + 1, good_neighbor_count - 1)])));

        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_feature_vector = this->feature_by_index(neighbor_index);
        const auto neighbor_distance = COMPARATOR::compare(query, neighbor_feature_vector, dist_func_param);
             
        // check the neighborhood of this node later, if its good enough
        if (neighbor_distance <= r * (1 + eps)) {
            next_nodes.emplace(neighbor_index, neighbor_distance);

          // remember the node, if its better than the worst in the result list
          if (neighbor_distance < r) {
            results.emplace(neighbor_index, neighbor_distance);

            // update the search radius
            if (results.size() > k) {
              results.pop();
              r = results.top().getDistance();
            }
          }
        }
      }
    }

    return results;
  }
};

/**
 * Load the graph
 */
auto load_sizebounded_graph(const char* path_graph, uint32_t new_max_size = 0)
{
  std::error_code ec{};
  auto file_size = std::filesystem::file_size(path_graph, ec);
  if (ec != std::error_code{})
  {
    fmt::print(stderr, "error when accessing test file, size is: {} message: {} \n", file_size, ec.message());
    perror("");
    abort();
  }

  auto ifstream = std::ifstream(path_graph, std::ios::binary);
  if (!ifstream.is_open())
  {
    fmt::print(stderr, "could not open {}\n", path_graph);
    perror("");
    abort();
  }

  // create feature space
  uint8_t data_type;
  ifstream.read(reinterpret_cast<char*>(&data_type), sizeof(data_type));
  uint16_t dim;
  ifstream.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  const auto feature_space = deglib::L2Space(dim);

  // create the graph
  uint32_t size;
  ifstream.read(reinterpret_cast<char*>(&size), sizeof(size));
  uint8_t edges_per_node;
  ifstream.read(reinterpret_cast<char*>(&edges_per_node), sizeof(edges_per_node));

  // if no new max size is set use the size of the graph from disk
  if(new_max_size == 0) 
    new_max_size = size;
  
  // if there is a max size is should be higher than the needed graph size from disk
  if(new_max_size < size) {
    fmt::print(stderr, "The graph in the {} file has {} nodes but the new max size is {}\n", path_graph, size, new_max_size);
    perror("");
    abort();
  }

  auto graph = deglib::graph::SizeBoundedGraph(new_max_size, edges_per_node, std::move(feature_space), ifstream, size);
  ifstream.close();

  return graph;
}

}  // namespace deglib::graph