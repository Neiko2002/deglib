#pragma once

#include <cstdint>
#include <limits>
#include <math.h>

#include <fmt/core.h>
#include <tsl/robin_map.h>

#include "deglib.h"
#include "distances.h"
#include "search.h"

namespace deglib::graph
{

/**
 * A immutable simple undirected n-regular graph. This version is prefered 
 * to use for existing graphs where only the search performance is important.
 * 
 * The node count and number of edges per nodes is known at construction time.
 * While the content of a node can be mutated after construction, it is not 
 * recommended. See SizeBoundedGraphs for a mutable version or understand the 
 * inner workings of the search function and memory layout, to make safe changes. 
 * The graph is n-regular where n is the number of eddes per node.
 * 
 * Furthermode the graph is undirected, if there is connection from A to B than 
 * there musst be one from B to A. All connections are stored in the neighbor 
 * indizies list of every node. The indizies are based on the indizies of their 
 * corresponding nodes. Each node has an index and an external label. The index 
 * is for internal computation and goes from 0 to the number of nodes. Where 
 * the external label can be any signed 32-bit integer.
 * 
 * The number of nodes is limited to uint32.max
 */
class ReadOnlyGraph : public deglib::search::SearchGraph {


  using SEARCHFUNC = deglib::search::ResultSet (*)(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count);

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext16(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float16Ext, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext8(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float8Ext, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext4(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float4Ext, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext16Residual(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float16ExtResiduals, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext4Residual(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float4ExtResiduals, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProduct(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt16(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat16Ext, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt8(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat8Ext, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt4(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat4Ext, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt16Residual(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat16ExtResiduals, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt4Residual(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat4ExtResiduals, use_max_distance_count>(entry_node_indizies, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static SEARCHFUNC getSearchFunction(const deglib::FloatSpace& feature_space) {
    const auto dim = feature_space.dim();
    const auto metric = feature_space.metric();

    if(metric == deglib::Metric::L2) {
      if (dim % 16 == 0)
        return deglib::graph::ReadOnlyGraph::searchL2Ext16<use_max_distance_count>;
      else if (dim % 8 == 0)
        return deglib::graph::ReadOnlyGraph::searchL2Ext8<use_max_distance_count>;
      else if (dim % 4 == 0)
        return deglib::graph::ReadOnlyGraph::searchL2Ext4<use_max_distance_count>;
      else if (dim > 16)
        return deglib::graph::ReadOnlyGraph::searchL2Ext16Residual<use_max_distance_count>;
      else if (dim > 4)
        return deglib::graph::ReadOnlyGraph::searchL2Ext4Residual<use_max_distance_count>;
    }
    else if(metric == deglib::Metric::InnerProduct)
    {

      if (dim % 16 == 0)
        return deglib::graph::ReadOnlyGraph::searchInnerProductExt16<use_max_distance_count>;
      else if (dim % 8 == 0)
        return deglib::graph::ReadOnlyGraph::searchInnerProductExt8<use_max_distance_count>;
      else if (dim % 4 == 0)
        return deglib::graph::ReadOnlyGraph::searchInnerProductExt4<use_max_distance_count>;
      else if (dim > 16)
        return deglib::graph::ReadOnlyGraph::searchInnerProductExt16Residual<use_max_distance_count>;
      else if (dim > 4)
        return deglib::graph::ReadOnlyGraph::searchInnerProductExt4Residual<use_max_distance_count>;
      else
        return deglib::graph::ReadOnlyGraph::searchInnerProduct<use_max_distance_count>;
    }
    return deglib::graph::ReadOnlyGraph::searchL2<use_max_distance_count>;
  }


  using EXPLOREFUNC = deglib::search::ResultSet (*)(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count);

  inline static deglib::search::ResultSet exploreL2(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext16(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float16Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext8(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float8Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext4(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float4Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext16Residual(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float16ExtResiduals>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext4Residual(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float4ExtResiduals>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProduct(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt16(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat16Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt8(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat8Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt4(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat4Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt16Residual(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat16ExtResiduals>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt4Residual(const ReadOnlyGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat4ExtResiduals>(entry_node_index, k, max_distance_computation_count);
  }

  inline static EXPLOREFUNC getExploreFunction(const deglib::FloatSpace& feature_space) {
    const auto dim = feature_space.dim();
    const auto metric = feature_space.metric();

    if(metric == deglib::Metric::L2) {
      if (dim % 16 == 0)
        return deglib::graph::ReadOnlyGraph::exploreL2Ext16;
      else if (dim % 8 == 0)
        return deglib::graph::ReadOnlyGraph::exploreL2Ext8;
      else if (dim % 4 == 0)
        return deglib::graph::ReadOnlyGraph::exploreL2Ext4;
      else if (dim > 16)
        return deglib::graph::ReadOnlyGraph::exploreL2Ext16Residual;
      else if (dim > 4)
        return deglib::graph::ReadOnlyGraph::exploreL2Ext4Residual;
    }
    else if(metric == deglib::Metric::InnerProduct)
    {

      if (dim % 16 == 0)
        return deglib::graph::ReadOnlyGraph::exploreInnerProductExt16;
      else if (dim % 8 == 0)
        return deglib::graph::ReadOnlyGraph::exploreInnerProductExt8;
      else if (dim % 4 == 0)
        return deglib::graph::ReadOnlyGraph::exploreInnerProductExt4;
      else if (dim > 16)
        return deglib::graph::ReadOnlyGraph::exploreInnerProductExt16Residual;
      else if (dim > 4)
        return deglib::graph::ReadOnlyGraph::exploreInnerProductExt4Residual;
      else
        return deglib::graph::ReadOnlyGraph::exploreInnerProduct;
    }

    return deglib::graph::ReadOnlyGraph::exploreL2;      
  }


  static uint32_t compute_aligned_byte_size_per_node(const uint8_t edges_per_node, const uint16_t feature_byte_size, const uint8_t alignment) {
      const uint32_t byte_size = uint32_t(feature_byte_size) + uint32_t(edges_per_node) * sizeof(uint32_t) + sizeof(uint32_t);
    if (alignment == 0)
      return  byte_size;
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
  static const uint8_t object_alignment = 16; 

  const uint32_t max_node_count_;
  const uint8_t edges_per_node_;
  const uint16_t feature_byte_size_;

  const uint32_t byte_size_per_node_;
  const uint32_t neighbor_indizies_offset_;
  const uint32_t external_label_offset_;

  // list of nodes (node: feature vector, indizies of neighbor nodes, external label)
  std::unique_ptr<std::byte[]> nodes_;
  std::byte* nodes_memory_;

  // map from the label of a node to the internal node index
  tsl::robin_map<uint32_t, uint32_t> label_to_index_;

  // internal search function with embedded distances function
  const SEARCHFUNC search_func_;
  const EXPLOREFUNC explore_func_;

  // distance calculation function between feature vectors of two graph nodes
  const deglib::FloatSpace feature_space_;

public:
  ReadOnlyGraph(const uint32_t max_node_count, const uint8_t edges_per_node, const deglib::FloatSpace feature_space)
      : edges_per_node_(edges_per_node), 
        max_node_count_(max_node_count), 
        feature_space_(feature_space),
        search_func_(getSearchFunction(feature_space)), explore_func_(getExploreFunction(feature_space)),
        feature_byte_size_(uint16_t(feature_space.get_data_size())), 
        byte_size_per_node_(compute_aligned_byte_size_per_node(edges_per_node, uint16_t(feature_space.get_data_size()), object_alignment)), 
        neighbor_indizies_offset_(uint32_t(feature_space.get_data_size())), 
        external_label_offset_(neighbor_indizies_offset_ + uint32_t(edges_per_node) * sizeof(uint32_t)), 
        nodes_(std::make_unique<std::byte[]>(size_t(max_node_count) * byte_size_per_node_ + object_alignment)), 
        nodes_memory_(compute_aligned_pointer(nodes_, object_alignment)), 
        label_to_index_(max_node_count) {
  }

  /**
   *  Load from file
   */
  ReadOnlyGraph(const uint32_t max_node_count, const uint8_t edges_per_node, const deglib::FloatSpace feature_space, std::ifstream& ifstream)
      : ReadOnlyGraph(max_node_count, edges_per_node, feature_space) {

    // copy the old data over
    uint32_t node_without_external = uint32_t(feature_space.get_data_size()) + uint32_t(edges_per_node) * sizeof(uint32_t);
    for (uint32_t i = 0; i < max_node_count; i++) {
      auto node = reinterpret_cast<char*>(this->node_by_index(i));
      ifstream.read(node, node_without_external);                     // read the feature vector and neighbor indizies
      ifstream.ignore(uint32_t(edges_per_node) * sizeof(float));      // skip the weights
      ifstream.read(node + node_without_external, sizeof(uint32_t));  // read the external label
      label_to_index_.emplace(this->getExternalLabel(i), i);
    }
  }

  /**
   * Current maximal capacity of nodes
   */ 
  const auto capacity() const {
    return max_node_count_;
  }

  /**
   * Number of nodes in the graph
   */
  const uint32_t size() const override {
    return (uint32_t) label_to_index_.size();
  }

  /**
   * Number of edges per node 
   */
  const uint8_t getEdgesPerNode() const override {
    return edges_per_node_;
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
    fmt::print(stderr, "Storing a readonly_graph {} is not possible\n", path_to_graph);
    perror("");
    abort();
  }

  /**
   * Add a new node. The neighbor indizies and feature vectors will be copied.
   */
  void addNode(const uint32_t external_label, const std::byte* feature_vector, const uint32_t* neighbor_indizies) {
    const auto new_internal_index = static_cast<uint32_t>(label_to_index_.size());
    label_to_index_.emplace(external_label, new_internal_index);

    auto node_memory = node_by_index(new_internal_index);
    std::memcpy(node_memory, feature_vector, feature_byte_size_);
    std::memcpy(node_memory + neighbor_indizies_offset_, neighbor_indizies, uint32_t(edges_per_node_) * 4);
    std::memcpy(node_memory + external_label_offset_, &external_label, 4);
  }

  /**
   * Performan a search but stops when the to_node was found.
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

    return std::vector<deglib::search::ObjectDistance>();
  }

  /**
   * The result set contains internal indizies. 
   */
  deglib::search::ResultSet search(const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) const override
  {
    if(max_distance_computation_count == 0)
      return search_func_(*this, entry_node_indizies, query, eps, k, 0);
    else {
      const auto limited_search_func = getSearchFunction<true>(this->feature_space_);
      return limited_search_func(*this, entry_node_indizies, query, eps, k, max_distance_computation_count);
    }
  }

  /**
   * The result set contains internal indizies. 
   */
  template <typename COMPARATOR, bool use_max_distance_count>
  deglib::search::ResultSet searchImpl(const std::vector<uint32_t>& entry_node_indizies, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count) const
  {
    const auto dist_func_param = this->feature_space_.get_dist_func_param();
    uint32_t distance_computation_count = 0;

    // set of checked node ids
    auto checked_ids = std::vector<bool>(this->size());

    // items to traverse next
    auto next_nodes = deglib::search::UncheckedSet();
    next_nodes.reserve(k*this->edges_per_node_);

    // result set
    // TODO: custom priority queue with an internal Variable Length Array wrapped in a macro with linear-scan search and memcopy 
    auto results = deglib::search::ResultSet();
    results.reserve(k);

    // copy the initial entry nodes and their distances to the query into the three containers
    for (auto&& index : entry_node_indizies) {
      if(checked_ids[index] == false) {
        checked_ids[index] = true;

        const auto feature = reinterpret_cast<const float*>(this->feature_by_index(index));
        const auto distance = COMPARATOR::compare(query, feature, dist_func_param);
        next_nodes.emplace(index, distance);
        results.emplace(index, distance);

        // early stop after to many computations
        if constexpr (use_max_distance_count) {
          if(distance_computation_count++ >= max_distance_computation_count)
            return results;
        }
      }
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

        // early stop after to many computations
        if constexpr (use_max_distance_count) {
          if(distance_computation_count++ >= max_distance_computation_count)
            return results;
        }
      }
    }

    return results;
  }

  
  /**
   * The result set contains internal indizies. 
   */
  deglib::search::ResultSet explore(const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count) const override
  {
    return explore_func_(*this, entry_node_index, k, max_distance_computation_count);
  }

  /**
   * The result set contains internal indizies. 
   */
  template <typename COMPARATOR>
  deglib::search::ResultSet exploreImpl(const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count) const
  {
    uint32_t distance_computation_count = 0;
    const auto dist_func_param = this->feature_space_.get_dist_func_param();

    // set of checked node ids
    auto checked_ids = std::vector<bool>(this->size());

    // items to traverse next
    auto next_nodes = deglib::search::UncheckedSet();
    next_nodes.reserve(k*this->edges_per_node_);

    // result set
    auto results = deglib::search::ResultSet();   
    results.reserve(k);

    // add the entry node index to the nodes which gets checked next and ignore it for further checks
    checked_ids[entry_node_index] = true;
    next_nodes.emplace(entry_node_index, 0);
    const auto query = this->feature_by_index(entry_node_index);

    // search radius
    auto r = std::numeric_limits<float>::max();

    // iterate as long as good elements are in the next_nodes queue and max_calcs is not yet reached
    auto good_neighbors = std::array<uint32_t, 256>();    // this limits the neighbor count to 256 using Variable Length Array wrapped in a macro
    while (next_nodes.empty() == false)
    {
      // next node to check
      const auto next_node = next_nodes.top();
      next_nodes.pop();

      uint8_t good_neighbor_count = 0;
      const auto neighbor_indizies = this->neighbors_by_index(next_node.getInternalIndex());
      for (uint8_t i = 0; i < this->edges_per_node_; i++) {
        const auto neighbor_index = neighbor_indizies[i];
        if (checked_ids[neighbor_index] == false)  {
          checked_ids[neighbor_index] = true;
          good_neighbors[good_neighbor_count++] = neighbor_index;
        }
      }

      if (good_neighbor_count == 0)
        continue;

      MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[0])));
      for (uint8_t i = 0; i < good_neighbor_count; i++) {
        MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[std::min(i + 1, good_neighbor_count - 1)])));

        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_feature_vector = this->feature_by_index(neighbor_index);
        const auto neighbor_distance = COMPARATOR::compare(query, neighbor_feature_vector, dist_func_param);

        if (neighbor_distance < r) {

          // check the neighborhood of this node later
          next_nodes.emplace(neighbor_index, neighbor_distance);

          // remember the node, if its better than the worst in the result list
          results.emplace(neighbor_index, neighbor_distance);

          // update the search radius
          if (results.size() > k) {
            results.pop();
            r = results.top().getDistance();
          }
        }

        // early stop after to many computations
        if(distance_computation_count++ >= max_distance_computation_count)
          return results;
      }
    }

    return results;
  }  

};


/**
 * Load the graph
 */
auto load_readonly_graph(const char* path_graph)
{
  std::error_code ec{};
  auto file_size = std::filesystem::file_size(path_graph, ec);
  if (ec != std::error_code{})
  {
    fmt::print(stderr, "error when accessing graph file {}, size is: {} message: {} \n", path_graph, file_size, ec.message());
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
  uint8_t metric_type;
  ifstream.read(reinterpret_cast<char*>(&metric_type), sizeof(metric_type));
  uint16_t dim;
  ifstream.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  const auto feature_space = deglib::FloatSpace(dim, static_cast<deglib::Metric>(metric_type));
  
  // create the graph
  uint32_t size;
  ifstream.read(reinterpret_cast<char*>(&size), sizeof(size));
  uint8_t edges_per_node;
  ifstream.read(reinterpret_cast<char*>(&edges_per_node), sizeof(edges_per_node));

  auto graph = deglib::graph::ReadOnlyGraph(size, edges_per_node, std::move(feature_space), ifstream);
  ifstream.close();

  return graph;
}

}  // namespace deglib::graph