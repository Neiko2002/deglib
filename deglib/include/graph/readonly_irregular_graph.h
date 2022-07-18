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
 * indices list of every node. The indices are based on the indices of their 
 * corresponding nodes. Each node has an index and an external label. The index 
 * is for internal computation and goes from 0 to the number of nodes. Where 
 * the external label can be any signed 32-bit integer.
 * 
 * The number of nodes is limited to uint32.max
 */
class ReadOnlyIrregularGraph : public deglib::search::SearchGraph {

  class AddressDistance
  {
      uint64_t address_;
      float distance_;

    public:
      AddressDistance() {}

      AddressDistance(const uint64_t address, const float distance) : address_(address), distance_(distance) {}

      inline const uint64_t getAddress() const { 
        return address_; 
      }

      inline const float getDistance() const { 
        return distance_; 
      }

      inline bool operator==(const AddressDistance& o) const { 
        return (distance_ == o.distance_) && (address_ == o.address_); 
      }

      inline bool operator<(const AddressDistance& o) const {
        if (distance_ == o.distance_)
          return address_ < o.address_;
        else
          return distance_ < o.distance_;
      }

      inline bool operator>(const AddressDistance& o) const {
        if (distance_ == o.distance_)
          return address_ > o.address_;
        else
          return distance_ > o.distance_;
      }
  };


  using SEARCHFUNC = deglib::search::ResultSet (*)(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count);

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext16(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float16Ext, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext8(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float8Ext, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext4(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float4Ext, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext16Residual(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float16ExtResiduals, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext4Residual(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float4ExtResiduals, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProduct(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt16(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat16Ext, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt8(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat8Ext, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt4(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat4Ext, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt16Residual(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat16ExtResiduals, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt4Residual(const ReadOnlyIrregularGraph& graph, const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat4ExtResiduals, use_max_distance_count>(entry_node_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static SEARCHFUNC getSearchFunction(const deglib::FloatSpace& feature_space) {
    const auto dim = feature_space.dim();
    const auto metric = feature_space.metric();

    if(metric == deglib::Metric::L2) {
      if (dim % 16 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::searchL2Ext16<use_max_distance_count>;
      else if (dim % 8 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::searchL2Ext8<use_max_distance_count>;
      else if (dim % 4 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::searchL2Ext4<use_max_distance_count>;
      else if (dim > 16)
        return deglib::graph::ReadOnlyIrregularGraph::searchL2Ext16Residual<use_max_distance_count>;
      else if (dim > 4)
        return deglib::graph::ReadOnlyIrregularGraph::searchL2Ext4Residual<use_max_distance_count>;
    }
    else if(metric == deglib::Metric::InnerProduct)
    {

      if (dim % 16 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::searchInnerProductExt16<use_max_distance_count>;
      else if (dim % 8 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::searchInnerProductExt8<use_max_distance_count>;
      else if (dim % 4 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::searchInnerProductExt4<use_max_distance_count>;
      else if (dim > 16)
        return deglib::graph::ReadOnlyIrregularGraph::searchInnerProductExt16Residual<use_max_distance_count>;
      else if (dim > 4)
        return deglib::graph::ReadOnlyIrregularGraph::searchInnerProductExt4Residual<use_max_distance_count>;
      else
        return deglib::graph::ReadOnlyIrregularGraph::searchInnerProduct<use_max_distance_count>;
    }
    return deglib::graph::ReadOnlyIrregularGraph::searchL2<use_max_distance_count>;
  }


  using EXPLOREFUNC = deglib::search::ResultSet (*)(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count);

  inline static deglib::search::ResultSet exploreL2(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext16(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float16Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext8(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float8Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext4(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float4Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext16Residual(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float16ExtResiduals>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext4Residual(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float4ExtResiduals>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProduct(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt16(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat16Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt8(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat8Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt4(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat4Ext>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt16Residual(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat16ExtResiduals>(entry_node_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt4Residual(const ReadOnlyIrregularGraph& graph, const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat4ExtResiduals>(entry_node_index, k, max_distance_computation_count);
  }

  inline static EXPLOREFUNC getExploreFunction(const deglib::FloatSpace& feature_space) {
    const auto dim = feature_space.dim();
    const auto metric = feature_space.metric();

    if(metric == deglib::Metric::L2) {
      if (dim % 16 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::exploreL2Ext16;
      else if (dim % 8 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::exploreL2Ext8;
      else if (dim % 4 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::exploreL2Ext4;
      else if (dim > 16)
        return deglib::graph::ReadOnlyIrregularGraph::exploreL2Ext16Residual;
      else if (dim > 4)
        return deglib::graph::ReadOnlyIrregularGraph::exploreL2Ext4Residual;
    }
    else if(metric == deglib::Metric::InnerProduct)
    {

      if (dim % 16 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::exploreInnerProductExt16;
      else if (dim % 8 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::exploreInnerProductExt8;
      else if (dim % 4 == 0)
        return deglib::graph::ReadOnlyIrregularGraph::exploreInnerProductExt4;
      else if (dim > 16)
        return deglib::graph::ReadOnlyIrregularGraph::exploreInnerProductExt16Residual;
      else if (dim > 4)
        return deglib::graph::ReadOnlyIrregularGraph::exploreInnerProductExt4Residual;
      else
        return deglib::graph::ReadOnlyIrregularGraph::exploreInnerProduct;
    }

    return deglib::graph::ReadOnlyIrregularGraph::exploreL2;      
  }


  static uint32_t compute_aligned_byte_size_per_node(const uint8_t edges_per_node, const uint16_t feature_byte_size, const uint8_t alignment) {
    //                                     feature vector  +   vertex index   +  neihgbor size   +               neighbor indices              + label
    const uint32_t byte_size = uint32_t(feature_byte_size) + sizeof(uint32_t) + sizeof(uint32_t) + uint32_t(edges_per_node) * sizeof(uint32_t) + sizeof(uint32_t);
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

  // distance calculation function between feature vectors of two graph nodes
  const deglib::FloatSpace feature_space_;
  const uint16_t feature_byte_size_;

  const uint32_t max_node_count_;
  const uint8_t max_edges_per_node_;
  const uint32_t max_byte_size_per_node_;

  const uint32_t vertex_index_offset_;
  const uint32_t neighbor_size_offset_;
  const uint32_t neighbor_indices_offset_;

  // list of nodes (node: feature vector, vertex index, number of neighbors, indices of neighbor nodes, external label)
  std::unique_ptr<std::byte[]> nodes_;
  std::byte* nodes_memory_;
  std::vector<uint64_t> index_to_address_; 

  // map from the label of a node to the internal node index
  tsl::robin_map<uint32_t, uint32_t> label_to_index_;

  // internal search function with embedded distances function
  const SEARCHFUNC search_func_;
  const EXPLOREFUNC explore_func_;


public:
  ReadOnlyIrregularGraph(const uint32_t max_node_count, const uint8_t edges_per_node, const deglib::FloatSpace feature_space)
      : feature_space_(feature_space),
        feature_byte_size_(uint16_t(feature_space.get_data_size())), 
        max_node_count_(max_node_count), 
        max_edges_per_node_(edges_per_node), 
        max_byte_size_per_node_(compute_aligned_byte_size_per_node(edges_per_node, feature_byte_size_, object_alignment)), 
        vertex_index_offset_(feature_byte_size_),
        neighbor_size_offset_(vertex_index_offset_ + sizeof(uint32_t)), 
        neighbor_indices_offset_(neighbor_size_offset_ + sizeof(uint32_t)), 
        nodes_(std::make_unique<std::byte[]>(size_t(max_node_count) * max_byte_size_per_node_ + object_alignment)), 
        nodes_memory_(compute_aligned_pointer(nodes_, object_alignment)), 
        search_func_(getSearchFunction(feature_space)), 
        explore_func_(getExploreFunction(feature_space)),
        label_to_index_(max_node_count),
        index_to_address_(max_node_count) {
          index_to_address_.clear();
  }

  /**
   *  Load from file
   */
  ReadOnlyIrregularGraph(const uint32_t max_node_count, const uint8_t edges_per_node, const deglib::FloatSpace feature_space, std::ifstream& ifstream)
      : ReadOnlyIrregularGraph(max_node_count, edges_per_node, feature_space) {

    auto neighbor_indices = std::vector<uint32_t>(edges_per_node);
    
    // copy the old data over
    for (uint32_t i = 0; i < max_node_count; i++) {
      index_to_address_.emplace_back(next_address(i, this->object_alignment));
      auto node = reinterpret_cast<char*>(this->node_by_index(i));

      ifstream.read(node, feature_byte_size_);                                                              // read the feature vector
      ifstream.read(reinterpret_cast<char*>(neighbor_indices.data()), size_t(edges_per_node) * sizeof(uint32_t));   // read the neighbor indices

      //  TODO a single self-loop is still in the graph after this (std::lower_bound and memcpy might be better)
      const auto last = std::unique(neighbor_indices.begin(), neighbor_indices.end());
      const auto neighbor_size = uint32_t(last - neighbor_indices.begin());

      reinterpret_cast<uint32_t*>(node + vertex_index_offset_)[0] = i;
      reinterpret_cast<uint32_t*>(node + neighbor_size_offset_)[0] = neighbor_size;
      auto neighbors = reinterpret_cast<uint32_t*>(node + neighbor_indices_offset_);
      std::memcpy(neighbors, neighbor_indices.data(), neighbor_size * sizeof(uint32_t));


      auto node_label_offset = neighbor_indices_offset_ + neighbor_size * sizeof(uint32_t);
      ifstream.ignore(size_t(edges_per_node) * sizeof(float));  // skip the weights
      ifstream.read(node + node_label_offset, sizeof(uint32_t));  // read the external label
      label_to_index_.emplace(this->getExternalLabel(i), i);
    }

    // sorting all edges of a vertex by the value of the weights -> has no influcene on the ANNS speed, only on small explore problems
    // auto neighbors = std::vector<std::pair<uint32_t,float>>(edges_per_node);
    // auto neighbor_indices = std::vector<uint32_t>(edges_per_node);
    // auto neighbor_weights = std::vector<float>(edges_per_node);

    // uint32_t node_neighbor_size = uint32_t(edges_per_node) * sizeof(uint32_t);
    // for (uint32_t i = 0; i < max_node_count; i++) {
    //   auto node = reinterpret_cast<char*>(this->node_by_index(i));
    //   ifstream.read(node, feature_byte_size_);                                              // read the feature vector
    //   ifstream.read(reinterpret_cast<char*>(neighbor_indices.data()), node_neighbor_size);  // read the neighbor indices
    //   ifstream.read(reinterpret_cast<char*>(neighbor_weights.data()), node_neighbor_size);  // read the neighbor weights
    //   ifstream.read(node + external_label_offset_, sizeof(uint32_t));                       // read the external label
    //   label_to_index_.emplace(this->getExternalLabel(i), i);

    //   // copy neighbor data into neighbor vertex
    //   neighbors.clear();
    //   for (size_t e = 0; e < edges_per_node; e++) 
    //     neighbors.emplace_back(neighbor_indices[e], neighbor_weights[e]);

    //   // sort the edges by their weight values
    //   std::sort(neighbors.begin(), neighbors.end(), [](const auto& x, const auto& y){return x.second < y.second;});

    //   // write the graph
    //   auto node_adjacent_indices = reinterpret_cast<uint32_t*>(node + neighbor_indices_offset_);
    //   for (size_t e = 0; e < edges_per_node; e++) 
    //     node_adjacent_indices[e] = neighbors[e].first;
    // }
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
    return max_edges_per_node_;
  }

  const deglib::SpaceInterface<float>& getFeatureSpace() const override {
    return this->feature_space_;
  }

    
private:  
  inline const uint64_t next_address(const uint32_t internal_idx, const uint32_t alignment) const {
    const auto node_count = uint32_t(index_to_address_.size());

    if(internal_idx + 1 <= node_count) {
      fmt::print(stderr, "index {} already exists in the graph\n", internal_idx);
      perror("");
      abort();
    }

    if(node_count == 0)
      return uint64_t(0);

    const auto last_index = uint32_t(node_count-1);
    const auto last_address = index_to_address_[last_index];
    const auto last_neighbor_size = neighbor_size_by_index(last_index);
    auto next_address = last_address + neighbor_indices_offset_ + last_neighbor_size * uint32_t(sizeof(uint32_t)) + uint32_t(sizeof(uint32_t));

    if(alignment > 0)
      next_address = ((next_address + alignment - 1) / alignment) * alignment;
    return next_address;
  }

  inline std::byte* node_by_index(const uint32_t internal_idx) const {
    return nodes_memory_ + address_by_index(internal_idx);
  }
  
  inline uint64_t address_by_index(const uint32_t internal_idx) const {
    return index_to_address_[internal_idx];
  }

  inline const uint32_t label_by_index(const uint32_t internal_idx) const {
    const auto node = node_by_index(internal_idx);
    const auto neighbor_size = *reinterpret_cast<const int32_t*>(node + neighbor_size_offset_);
    return *reinterpret_cast<const int32_t*>(node + neighbor_indices_offset_ + neighbor_size * sizeof(uint32_t));
  }

  inline const std::byte* feature_by_index(const uint32_t internal_idx) const{
    return node_by_index(internal_idx);
  }

  inline const uint32_t* neighbors_by_index(const uint32_t internal_idx) const {
    return reinterpret_cast<uint32_t*>(node_by_index(internal_idx) + neighbor_size_offset_);
  }

  inline const uint32_t* neighbor_indices_by_index(const uint32_t internal_idx) const {
    return reinterpret_cast<uint32_t*>(node_by_index(internal_idx) + neighbor_indices_offset_);
  }

  inline const uint32_t neighbor_size_by_index(const uint32_t internal_idx) const {
    return *reinterpret_cast<const int32_t*>(node_by_index(internal_idx) + neighbor_size_offset_);
  }


  inline std::byte* node_by_address(const uint64_t address) const {
    return nodes_memory_ + address;
  }

  inline const std::byte* feature_by_address(const uint64_t address) const{
    return node_by_address(address);
  }

  inline const uint32_t index_by_address(const uint64_t address) const{
    return *reinterpret_cast<const int32_t*>(node_by_address(address) + vertex_index_offset_);
  }

  inline const uint32_t* neighbors_by_address(const uint64_t address) const {
    return reinterpret_cast<uint32_t*>(node_by_address(address) + neighbor_size_offset_);
  }

  inline const uint32_t* neighbor_indices_by_address(const uint64_t address) const {
    return reinterpret_cast<uint32_t*>(node_by_address(address) + neighbor_indices_offset_);
  }

  inline const uint32_t neighbor_size_by_address(const uint64_t address) const {
    return *reinterpret_cast<const int32_t*>(node_by_address(address) + neighbor_size_offset_);
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

  inline const uint32_t* getNeighborIndices(const uint32_t internal_idx) const override {
    return neighbor_indices_by_index(internal_idx);
  }

  // TODO needs override
  inline const uint32_t getNeighborSize(const uint32_t internal_idx) const {
    return neighbor_size_by_index(internal_idx);
  }

  inline const bool hasNode(const uint32_t external_label) const override {
    return label_to_index_.contains(external_label);
  }

  // TODO make everything const
  inline const bool hasEdge(const uint32_t internal_index, const uint32_t neighbor_index) const override {
    auto neighbor_size = neighbor_size_by_index(internal_index);
    auto neighbor_indices = neighbor_indices_by_index(internal_index);
    auto neighbor_indices_end = neighbor_indices + neighbor_size;  
    auto neighbor_ptr = std::lower_bound(neighbor_indices, neighbor_indices_end, neighbor_index); 
    return (*neighbor_ptr == neighbor_index);
  }

  const bool saveGraph(const char* path_to_graph) const override {
    fmt::print(stderr, "Storing a readonly_graph {} is not possible\n", path_to_graph);
    perror("");
    abort();
  }

  /**
   * Performan a search but stops when the to_node was found.
   */
  std::vector<deglib::search::ObjectDistance> hasPath(const std::vector<uint32_t>& entry_node_indices, const uint32_t to_node, const float eps, const uint32_t k) const override
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
    for (auto&& index : entry_node_indices) {
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
      const auto neighbors = this->neighbors_by_index(next_node.getInternalIndex());
      const auto neighbor_size = neighbors[0];
      const auto neighbor_indices = neighbors+1;
      for (size_t i = 0; i < neighbor_size; i++) {
        const auto neighbor_index = neighbor_indices[i];

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
   * The result set contains internal indices. 
   */
  deglib::search::ResultSet search(const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) const override
  {
    if(max_distance_computation_count == 0)
      return search_func_(*this, entry_node_indices, query, eps, k, 0);
    else {
      const auto limited_search_func = getSearchFunction<true>(this->feature_space_);
      return limited_search_func(*this, entry_node_indices, query, eps, k, max_distance_computation_count);
    }
  }

  /**
   * The result set contains internal indices. 
   */
  template <typename COMPARATOR, bool use_max_distance_count>
  deglib::search::ResultSet searchImpl(const std::vector<uint32_t>& entry_node_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count) const
  {
    const auto dist_func_param = this->feature_space_.get_dist_func_param();
    uint32_t distance_computation_count = 0;

    // set of checked node ids
    auto checked_ids = std::vector<bool>(this->size());

    // items to traverse next
    auto next_nodes = deglib::search::PQV<std::greater<AddressDistance>, AddressDistance>();
    next_nodes.reserve(k*this->max_edges_per_node_);

    // result set
    // TODO: custom priority queue with an internal Variable Length Array wrapped in a macro with linear-scan search and memcopy 
    auto results = deglib::search::ResultSet();
    results.reserve(k);

    // copy the initial entry nodes and their distances to the query into the three containers
    for (auto&& index : entry_node_indices) {
      if(checked_ids[index] == false) {
        checked_ids[index] = true;

        const auto feature = reinterpret_cast<const float*>(this->feature_by_index(index));
        const auto distance = COMPARATOR::compare(query, feature, dist_func_param);
        next_nodes.emplace(address_by_index(index), distance);
        results.emplace(index, distance);

        // early stop after to many computations
        if constexpr (use_max_distance_count) {
          if(++distance_computation_count >= max_distance_computation_count)
            return results;
        }
      }
    }

    // search radius
    auto r = std::numeric_limits<float>::max();

    // iterate as long as good elements are in the next_nodes queue     
    auto good_neighbor_addresses = std::array<uint64_t, 256>();    // this limits the neighbor count to 256 using Variable Length Array wrapped in a macro
    while (next_nodes.empty() == false)
    {
      // next node to check
      const auto next_node = next_nodes.top();
      next_nodes.pop();

      // max distance reached
      if (next_node.getDistance() > r * (1 + eps)) 
        break;

      uint32_t good_neighbor_count = 0;
      const auto neighbors = this->neighbors_by_address(next_node.getAddress());
      const auto neighbor_size = neighbors[0];
      const auto neighbor_indices = neighbors+1;
      for (uint32_t i = 0; i < neighbor_size; i++) {
        const auto neighbor_index = neighbor_indices[i];
        if (checked_ids[neighbor_index] == false)  {
          checked_ids[neighbor_index] = true;
          good_neighbor_addresses[good_neighbor_count++] = this->address_by_index(neighbor_index);
        }
      }

      if (good_neighbor_count == 0)
        continue;

      MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_address(good_neighbor_addresses[0])));
      for (uint32_t i = 0; i < good_neighbor_count; i++) {
        MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_address(good_neighbor_addresses[std::min(i + 1, good_neighbor_count - 1)])));

        const auto neighbor_address = good_neighbor_addresses[i];
        const auto neighbor_feature_vector = this->feature_by_address(neighbor_address);
        const auto neighbor_distance = COMPARATOR::compare(query, neighbor_feature_vector, dist_func_param);
             
        // check the neighborhood of this node later, if its good enough
        if (neighbor_distance <= r * (1 + eps)) {
          next_nodes.emplace(neighbor_address, neighbor_distance);

          // remember the node, if its better than the worst in the result list
          if (neighbor_distance < r) {
            results.emplace(index_by_address(neighbor_address), neighbor_distance);

            // update the search radius
            if (results.size() > k) {
              results.pop();
              r = results.top().getDistance();
            }
          }
        }

        // early stop after to many computations
        if constexpr (use_max_distance_count) {
          if(++distance_computation_count >= max_distance_computation_count)
            return results;
        }
      }
    }

    return results;
  }

  
  /**
   * The result set contains internal indices. 
   */
  deglib::search::ResultSet explore(const uint32_t entry_node_index, const uint32_t k, const uint32_t max_distance_computation_count) const override
  {
    return explore_func_(*this, entry_node_index, k, max_distance_computation_count);
  }

  /**
   * The result set contains internal indices. 
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
    next_nodes.reserve(k*this->max_edges_per_node_);

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
      const auto neighbor_size = this->neighbor_size_by_index(next_node.getInternalIndex());
      const auto neighbor_indices = this->neighbor_indices_by_index(next_node.getInternalIndex());
      for (uint8_t i = 0; i < neighbor_size; i++) {
        const auto neighbor_index = neighbor_indices[i];
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
        if(++distance_computation_count >= max_distance_computation_count)
          return results;
      }
    }

    return results;
  }  

};


/**
 * Load the graph
 */
auto load_readonly_irregular_graph(const char* path_graph)
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

  auto graph = deglib::graph::ReadOnlyIrregularGraph(size, edges_per_node, std::move(feature_space), ifstream);
  ifstream.close();

  return graph;
}

}  // namespace deglib::graph