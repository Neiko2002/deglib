#pragma once

#include <cstdint>
#include <limits>
#include <queue>
#include <math.h>
#include <span>

#include <fmt/core.h>
#include <tsl/robin_hash.h>
#include <tsl/robin_map.h>

#include "graph.h"
#include "repository.h"
#include "search.h"
#include "memory.h"

namespace deglib::graph
{

/**
 * A size bounded undirected and weighted n-regular graph.
 * 
 * The vertex count and number of edges per vertices is bounded to a fixed value at 
 * construction time. The graph is therefore n-regular where n is the number of 
 * eddes per vertex.
 * 
 * Furthermode the graph is undirected, if there is connection from A to B than 
 * there musst be one from B to A. All connections are stored in the neighbor 
 * indices list of every vertex. The indices are based on the indices of their 
 * corresponding vertices. Each vertex has an index and an external label. The index 
 * is for internal computation and goes from 0 to the number of vertices. Where 
 * the external label can be any signed 32-bit integer. The indices in the 
 * neighbors list are ascending sorted.
 * 
 * Every edge contains of a neighbor vertex index and a weight. The weights and
 * neighbor indices are in separated list, but have the same order.
 * 
 * The number of vertices is limited to uint32.max
 */
class SizeBoundedGraph : public deglib::graph::MutableGraph {

  using SEARCHFUNC = deglib::search::ResultSet (*)(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count);

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext16(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float16Ext, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext8(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float8Ext, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext4(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float4Ext, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext16Residual(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float16ExtResiduals, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchL2Ext4Residual(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::L2Float4ExtResiduals, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProduct(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt16(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat16Ext, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt8(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat8Ext, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt4(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat4Ext, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt16Residual(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat16ExtResiduals, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static deglib::search::ResultSet searchInnerProductExt4Residual(const SizeBoundedGraph& graph, const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.searchImpl<deglib::distances::InnerProductFloat4ExtResiduals, use_max_distance_count>(entry_vertex_indices, query, eps, k, max_distance_computation_count);
  }

  template <bool use_max_distance_count = false>
  inline static SEARCHFUNC getSearchFunction(const deglib::FloatSpace& feature_space) {
    const auto dim = feature_space.dim();
    const auto metric = feature_space.metric();

    if(metric == deglib::Metric::L2) {
      if (dim % 16 == 0)
        return deglib::graph::SizeBoundedGraph::searchL2Ext16<use_max_distance_count>;
      else if (dim % 8 == 0)
        return deglib::graph::SizeBoundedGraph::searchL2Ext8<use_max_distance_count>;
      else if (dim % 4 == 0)
        return deglib::graph::SizeBoundedGraph::searchL2Ext4<use_max_distance_count>;
      else if (dim > 16)
        return deglib::graph::SizeBoundedGraph::searchL2Ext16Residual<use_max_distance_count>;
      else if (dim > 4)
        return deglib::graph::SizeBoundedGraph::searchL2Ext4Residual<use_max_distance_count>;
    }
    else if(metric == deglib::Metric::InnerProduct)
    {

      if (dim % 16 == 0)
        return deglib::graph::SizeBoundedGraph::searchInnerProductExt16<use_max_distance_count>;
      else if (dim % 8 == 0)
        return deglib::graph::SizeBoundedGraph::searchInnerProductExt8<use_max_distance_count>;
      else if (dim % 4 == 0)
        return deglib::graph::SizeBoundedGraph::searchInnerProductExt4<use_max_distance_count>;
      else if (dim > 16)
        return deglib::graph::SizeBoundedGraph::searchInnerProductExt16Residual<use_max_distance_count>;
      else if (dim > 4)
        return deglib::graph::SizeBoundedGraph::searchInnerProductExt4Residual<use_max_distance_count>;
      else
        return deglib::graph::SizeBoundedGraph::searchInnerProduct<use_max_distance_count>;
    }
    return deglib::graph::SizeBoundedGraph::searchL2<use_max_distance_count>;
  }


  using EXPLOREFUNC = deglib::search::ResultSet (*)(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count);

  inline static deglib::search::ResultSet exploreL2(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext16(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float16Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext8(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float8Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext4(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float4Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext16Residual(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float16ExtResiduals>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreL2Ext4Residual(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::L2Float4ExtResiduals>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProduct(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt16(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat16Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt8(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat8Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt4(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat4Ext>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt16Residual(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat16ExtResiduals>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static deglib::search::ResultSet exploreInnerProductExt4Residual(const SizeBoundedGraph& graph, const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count = 0) {
    return graph.exploreImpl<deglib::distances::InnerProductFloat4ExtResiduals>(entry_vertex_index, k, max_distance_computation_count);
  }

  inline static EXPLOREFUNC getExploreFunction(const deglib::FloatSpace& feature_space) {
    const auto dim = feature_space.dim();
    const auto metric = feature_space.metric();

    if(metric == deglib::Metric::L2) {
      if (dim % 16 == 0)
        return deglib::graph::SizeBoundedGraph::exploreL2Ext16;
      else if (dim % 8 == 0)
        return deglib::graph::SizeBoundedGraph::exploreL2Ext8;
      else if (dim % 4 == 0)
        return deglib::graph::SizeBoundedGraph::exploreL2Ext4;
      else if (dim > 16)
        return deglib::graph::SizeBoundedGraph::exploreL2Ext16Residual;
      else if (dim > 4)
        return deglib::graph::SizeBoundedGraph::exploreL2Ext4Residual;
    }
    else if(metric == deglib::Metric::InnerProduct)
    {

      if (dim % 16 == 0)
        return deglib::graph::SizeBoundedGraph::exploreInnerProductExt16;
      else if (dim % 8 == 0)
        return deglib::graph::SizeBoundedGraph::exploreInnerProductExt8;
      else if (dim % 4 == 0)
        return deglib::graph::SizeBoundedGraph::exploreInnerProductExt4;
      else if (dim > 16)
        return deglib::graph::SizeBoundedGraph::exploreInnerProductExt16Residual;
      else if (dim > 4)
        return deglib::graph::SizeBoundedGraph::exploreInnerProductExt4Residual;
      else
        return deglib::graph::SizeBoundedGraph::exploreInnerProduct;
    }

    return deglib::graph::SizeBoundedGraph::exploreL2;      
  }

  static uint32_t compute_aligned_byte_size_per_vertex(const uint8_t edges_per_vertex, const uint16_t feature_byte_size, const uint8_t alignment) {
    const uint32_t byte_size = uint32_t(feature_byte_size) + uint32_t(edges_per_vertex) * (sizeof(uint32_t) + sizeof(float)) + sizeof(uint32_t);
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

  // alignment of vertex information in bytes (all feature vectors will be 256bit aligned for faster SIMD processing)
  static const uint8_t object_alignment = 32; 

  const uint32_t max_vertex_count_;
  const uint8_t edges_per_vertex_;
  const uint16_t feature_byte_size_;

  const uint32_t byte_size_per_vertex_;
  const uint32_t neighbor_indices_offset_;
  const uint32_t neighbor_weights_offset_;
  const uint32_t external_label_offset_;

  // list of vertices (vertex: std::byte* feature vector, uint32_t* indices of neighbor vertices, float* weights of neighbor vertices, uint32_t external label)      
  std::unique_ptr<std::byte[]> vertices_;
  std::byte* vertices_memory_;

  // map from the label of a vertex to the internal vertex index
  tsl::robin_map<uint32_t, uint32_t> label_to_index_;

  // internal search function with embedded distances function
  const SEARCHFUNC search_func_;
  const EXPLOREFUNC explore_func_;

  // distance calculation function between feature vectors of two graph vertices
  const deglib::FloatSpace feature_space_;

 public:
  SizeBoundedGraph(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::FloatSpace feature_space)
      : edges_per_vertex_(edges_per_vertex), 
        max_vertex_count_(max_vertex_count), 
        feature_space_(feature_space),
        search_func_(getSearchFunction(feature_space)), explore_func_(getExploreFunction(feature_space)),
        feature_byte_size_(uint16_t(feature_space.get_data_size())), 
        byte_size_per_vertex_(compute_aligned_byte_size_per_vertex(edges_per_vertex, uint16_t(feature_space.get_data_size()), object_alignment)), 
        neighbor_indices_offset_(uint32_t(feature_space.get_data_size())),
        neighbor_weights_offset_(neighbor_indices_offset_ + uint32_t(edges_per_vertex) * sizeof(uint32_t)),
        external_label_offset_(neighbor_weights_offset_ + uint32_t(edges_per_vertex) * sizeof(float)), 
        vertices_(std::make_unique<std::byte[]>(size_t(max_vertex_count) * byte_size_per_vertex_ + object_alignment)), 
        vertices_memory_(compute_aligned_pointer(vertices_, object_alignment)), 
        label_to_index_(max_vertex_count) {
  }

  /**
   *  Load from file
   */
  SizeBoundedGraph(const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::FloatSpace feature_space, std::ifstream& ifstream, const uint32_t size)
      : SizeBoundedGraph(max_vertex_count, edges_per_vertex, std::move(feature_space)) {

    // copy the old data over
    uint32_t file_byte_size_per_vertex = compute_aligned_byte_size_per_vertex(this->edges_per_vertex_, this->feature_byte_size_, 0);
    for (uint32_t i = 0; i < size; i++) {
      ifstream.read(reinterpret_cast<char*>(this->vertex_by_index(i)), file_byte_size_per_vertex);
      label_to_index_.emplace(this->getExternalLabel(i), i);
    }
  }

  /**
   * Current maximal capacity of vertices
   */ 
  const auto capacity() const {
    return this->max_vertex_count_;
  }

  /**
   * Number of vertices in the graph
   */
  const uint32_t size() const final {
    return (uint32_t) this->label_to_index_.size();
  }

  /**
   * Number of edges per vertex 
   */
  const uint8_t getEdgesPerNode() const final {
    return this->edges_per_vertex_;
  }

  const deglib::SpaceInterface<float>& getFeatureSpace() const final {
    return this->feature_space_;
  }

private:  
  inline std::byte* vertex_by_index(const uint32_t internal_idx) const {
    return vertices_memory_ + size_t(internal_idx) * byte_size_per_vertex_;
  }

  inline const uint32_t label_by_index(const uint32_t internal_idx) const {
    return *reinterpret_cast<const int32_t*>(vertex_by_index(internal_idx) + external_label_offset_);
  }

  inline const std::byte* feature_by_index(const uint32_t internal_idx) const{
    return vertex_by_index(internal_idx);
  }

  inline const uint32_t* neighbors_by_index(const uint32_t internal_idx) const {
    return reinterpret_cast<uint32_t*>(vertex_by_index(internal_idx) + neighbor_indices_offset_);
  }

  inline const float* weights_by_index(const uint32_t internal_idx) const {
    return reinterpret_cast<const float*>(vertex_by_index(internal_idx) + neighbor_weights_offset_);
  }

public:

  /**
   * convert an external label to an internal index
   */ 
  inline const uint32_t getInternalIndex(const uint32_t external_label) const final {
    return label_to_index_.find(external_label)->second;
  }

  inline const uint32_t getExternalLabel(const uint32_t internal_idx) const final {
    return label_by_index(internal_idx);
  }

  inline const std::byte* getFeatureVector(const uint32_t internal_idx) const final{
    return feature_by_index(internal_idx);
  }

  inline const uint32_t* getNeighborIndices(const uint32_t internal_idx) const final {
    return neighbors_by_index(internal_idx);
  }

  inline const float* getNeighborWeights(const uint32_t internal_idx) const final {
    return weights_by_index(internal_idx);
  }

  inline const float getEdgeWeight(const uint32_t internal_index, const uint32_t neighbor_index) const final {
    auto neighbor_indices = neighbors_by_index(internal_index);
    auto neighbor_indices_end = neighbor_indices + this->edges_per_vertex_;  
    auto neighbor_ptr = std::lower_bound(neighbor_indices, neighbor_indices_end, neighbor_index); 
    if(*neighbor_ptr == neighbor_index) {
      auto weight_index = neighbor_ptr - neighbor_indices;
      return weights_by_index(internal_index)[weight_index];
    }
    return -1;
  }

  inline const bool hasNode(const uint32_t external_label) const final {
    return label_to_index_.contains(external_label);
  }

  inline const bool hasEdge(const uint32_t internal_index, const uint32_t neighbor_index) const final {
    auto neighbor_indices = neighbors_by_index(internal_index);
    auto neighbor_indices_end = neighbor_indices + this->edges_per_vertex_;  
    return std::binary_search(neighbor_indices, neighbor_indices_end, neighbor_index);
  }

  bool reorderNodes(const std::vector<uint32_t> order_vector) final {
    const uint32_t vertex_count = this->size();
    const uint32_t bytes_per_vertex = this->byte_size_per_vertex_;
    const uint32_t edges_per_vertex = this->edges_per_vertex_;
    
    // check if every new vertex index is valid place in the graph
    if(order_vector.size() != vertex_count)
      return false;
    for (uint32_t i = 0; i < vertex_count; i++)
      if(order_vector[i] > vertex_count)
        return false;

    // copy the entire content into a new memory segment with a different order
    {
      auto new_vertices = std::make_unique<std::byte[]>(size_t(vertex_count) * bytes_per_vertex + object_alignment);
      auto new_vertices_ptr = new_vertices.get();
      auto old_vertices_ptr = this->vertices_.get();
      for (uint32_t from_index = 0; from_index < vertex_count; from_index++)  {
        auto from_offset = size_t(from_index) * bytes_per_vertex;
        auto to_offset = size_t(order_vector[from_index]) * bytes_per_vertex;
        std::memcpy(new_vertices_ptr + to_offset, old_vertices_ptr + from_offset, bytes_per_vertex);
      }
      std::memcpy(old_vertices_ptr, new_vertices_ptr, size_t(vertex_count) * bytes_per_vertex + object_alignment);
    }

    // change the internal_ids of every vertex
    auto neighbors = std::vector<std::pair<uint32_t,float>>(edges_per_vertex);
    auto neighbor_indices = std::vector<uint32_t>(edges_per_vertex);
    auto neighbor_weights = std::vector<float>(edges_per_vertex);
    for (uint32_t i = 0; i < vertex_count; i++) {
      neighbors.clear();
      neighbor_indices.clear();
      neighbor_weights.clear();

      // remap the old internal indices to the new indices
      auto old_neighbors = this->neighbors_by_index(i);
      auto old_weights = this->weights_by_index(i);
      for (uint32_t e = 0; e < edges_per_vertex; e++) 
        neighbors.emplace_back(order_vector[old_neighbors[e]], old_weights[e]);
      
      // sort the edges by their new index values
      std::sort(neighbors.begin(), neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
      neighbor_indices.clear();
      neighbor_weights.clear();
      for (auto &&neighbor : neighbors) {
        neighbor_indices.emplace_back(neighbor.first);
        neighbor_weights.emplace_back(neighbor.second);
      }
      
      // store the edges of the new vertex
      this->changeEdges(i, neighbor_indices.data(), neighbor_weights.data());
    }


    // index -> to index and moved
    // auto new_order = std::vector<std::pair<uint32_t,boolean>>(vertex_count);
    // for (size_t i = 0; i < vertex_count; i++) 
    //   new_order.emplace_back(order_vector[i], false);

    // // temporary 
    // std::unique_ptr<std::byte[]> buffer_vertex = std::make_unique<std::byte[]>(vertex_bytes);

    // // move the content of the vertices and remap the internal ids
    // for (uint32_t i = 0; i < vertex_count; i++)  {
    //   auto vertex_index = i;
    //   auto& vertex_order = new_order[vertex_index];
    //   if(vertex_order.second == false) {
    //     auto vertex = this->vertex_by_index(vertex_index);

    //     // make changes in place

    //     // make a copy of the target vertex
    //     auto to_vertex_index = vertex_order.first;
    //     auto to_vertex = this->vertex_by_index(to_vertex_index);

    //     std::memcpy(buffer_vertex.get(), to_vertex, vertex_bytes);
    //   }
    // }

    return false;
  }

  const bool saveGraph(const char* path_to_graph) const final {
    auto out = std::ofstream(path_to_graph, std::ios::out | std::ios::binary);

    // check open file for write
    if (!out.is_open()) {
      fmt::print(stderr, "Error in open file {}\n", path_to_graph);
      return false;
    }

    // store feature space information
    uint8_t metric_type = static_cast<uint8_t>(feature_space_.metric());
    out.write(reinterpret_cast<const char*>(&metric_type), sizeof(metric_type));
    uint16_t dim = uint16_t(this->feature_space_.dim());
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

    // store graph information
    uint32_t size = uint32_t(this->size());
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(&this->edges_per_vertex_), sizeof(this->edges_per_vertex_));

    // store the existing vertices
    uint32_t byte_size_per_vertex = compute_aligned_byte_size_per_vertex(this->edges_per_vertex_, this->feature_byte_size_, 0);
    for (uint32_t i = 0; i < size; i++)
      out.write(reinterpret_cast<const char*>(this->vertex_by_index(i)), byte_size_per_vertex);    
    out.close();

    return true;
  }

  /**
   * Add a new vertex. The neighbor indices will be prefilled with a self-loop, the weights will be 0.
   * 
   * @return the internal index of the new vertex
   */
  uint32_t addNode(const uint32_t external_label, const std::byte* feature_vector) final {
    const auto new_internal_index = static_cast<uint32_t>(label_to_index_.size());
    label_to_index_.emplace(external_label, new_internal_index);

    auto vertex_memory = vertex_by_index(new_internal_index);
    std::memcpy(vertex_memory, feature_vector, feature_byte_size_);
    std::fill_n(reinterpret_cast<uint32_t*>(vertex_memory + neighbor_indices_offset_), this->edges_per_vertex_, new_internal_index); // temporary self loop
    std::fill_n(reinterpret_cast<float*>(vertex_memory + neighbor_weights_offset_), this->edges_per_vertex_, float(0)); // 0 weight
    std::memcpy(vertex_memory + external_label_offset_, &external_label, sizeof(uint32_t));

    return new_internal_index;
  }

  /**
    * Remove an existing vertex.
    */
  std::vector<uint32_t> removeNode(const uint32_t external_label) final {
    const auto internal_index = getInternalIndex(external_label);
    const auto last_internal_index = static_cast<uint32_t>(this->label_to_index_.size() - 1);

    // since the last_internal_index will be moved to the internal_index, 
    // update the current neighbor list if the last_internal_index is present
    if(hasEdge(internal_index, last_internal_index)) {
      changeEdge(internal_index, last_internal_index, internal_index, 0);
      changeEdge(last_internal_index, internal_index, last_internal_index, 0);
    }

    // copy the neighbor list to return it later
    const auto neighbor_indices = neighbors_by_index(internal_index);
    const auto involved_indices = std::vector<uint32_t>(neighbor_indices, neighbor_indices + this->edges_per_vertex_);

    // replace all references to the internal_index with a self-reference of the corresponding vertex
    for (size_t index = 0; index < this->edges_per_vertex_; index++) 
      changeEdge(neighbor_indices[index], internal_index, neighbor_indices[index], 0);

    // the last index will be moved to the internal_index position and overwrite its content
    if(internal_index != last_internal_index) {

      // update the neighbor list of the last vertex to reflex its new vertex index
      const auto last_neighbor_indices = neighbors_by_index(last_internal_index);
      const auto last_neighbor_weights = weights_by_index(last_internal_index);
      for (size_t index = 0; index < this->edges_per_vertex_; index++) 
        changeEdge(last_neighbor_indices[index], last_internal_index, internal_index, last_neighbor_weights[index]);
      
      // copy the last vertex to the vertex which gets removed
      std::memcpy(vertex_by_index(internal_index), vertex_by_index(last_internal_index), this->byte_size_per_vertex_);

      // update the index position of the last label
      const auto last_label = label_by_index(last_internal_index);
      label_to_index_[last_label] = internal_index;
    }

    // remove the external label from the hash map
    label_to_index_.erase(external_label);

    // return all neighbors of the deleted vertex
    return involved_indices;
  }

  /**
   * Swap a neighbor with another neighbor and its weight.
   * 
   * @param internal_index vertex index which neighbors should be changed
   * @param from_neighbor_index neighbor index to remove
   * @param to_neighbor_index neighbor index to add
   * @param to_neighbor_weight weight of the neighbor to add
   * @return true if the from_neighbor_index was found and changed
   */
  bool changeEdge(const uint32_t internal_index, const uint32_t from_neighbor_index, const uint32_t to_neighbor_index, const float to_neighbor_weight) final {
    auto vertex_memory = vertex_by_index(internal_index);

    auto neighbor_indices = reinterpret_cast<uint32_t*>(vertex_memory + neighbor_indices_offset_);  // list of neighbor indizizes
    auto neighbor_indices_end = neighbor_indices + this->edges_per_vertex_;                         // end of the list
    auto from_ptr = std::lower_bound(neighbor_indices, neighbor_indices_end, from_neighbor_index);  // possible position of the from_neighbor_index in the neighbor list
  
    // from_neighbor_index not found in the neighbor list
    if(*from_ptr != from_neighbor_index) {
      fmt::print(stderr, "changeEdge: vertex {} does not have an edge to {} and therefore can not swap it with {} and with distance {}\n", internal_index, from_neighbor_index, to_neighbor_index, to_neighbor_weight);
      return false;
    }

    auto neighbor_weights = reinterpret_cast<float*>(vertex_memory + neighbor_weights_offset_);     // list of neighbor weights
    auto to_ptr = std::lower_bound(neighbor_indices, neighbor_indices_end, to_neighbor_index);      // neighbor in the list which has a lower index number than to_neighbor_index
    auto from_list_idx = uint32_t(from_ptr - neighbor_indices);                                     // index of the from_neighbor_index in the neighbor list
    auto to_list_idx = uint32_t(to_ptr - neighbor_indices);                                         // index where to place the to_neighbor_index 

    // Make same space before inserting the new values
    if(from_list_idx < to_list_idx) {
      std::memmove(neighbor_indices + from_list_idx, neighbor_indices + from_list_idx + 1, (to_list_idx - from_list_idx) * sizeof(uint32_t)); 
      std::memmove(neighbor_weights + from_list_idx, neighbor_weights + from_list_idx + 1, (to_list_idx - from_list_idx) * sizeof(float)); 
      to_list_idx--;
    } else if(to_list_idx < from_list_idx) {
      std::memmove(neighbor_indices + to_list_idx + 1, neighbor_indices + to_list_idx, (from_list_idx - to_list_idx) * sizeof(uint32_t));
      std::memmove(neighbor_weights + to_list_idx + 1, neighbor_weights + to_list_idx, (from_list_idx - to_list_idx) * sizeof(float));
    }

    auto idxs = std::span<uint32_t>(neighbor_indices, this->edges_per_vertex_);

    neighbor_indices[to_list_idx] = to_neighbor_index;
    neighbor_weights[to_list_idx] = to_neighbor_weight;

    return true;
  }

  /**
   * Change all edges of a vertex.
   * The neighbor indices und weights will be copied.
   * The neighbor array need to have enough neighbors to match the edge-per-vertex count of the graph.
   * The indices in the neighbor_indices array must be sorted.
   */
  void changeEdges(const uint32_t internal_index, const uint32_t* neighbor_indices, const float* neighbor_weights) final {
    auto vertex_memory = vertex_by_index(internal_index);
    std::memcpy(vertex_memory + neighbor_indices_offset_, neighbor_indices, uint32_t(edges_per_vertex_) * sizeof(uint32_t));
    std::memcpy(vertex_memory + neighbor_weights_offset_, neighbor_weights, uint32_t(edges_per_vertex_) * sizeof(float));
  }

  /**
   * Performan a search but stops when the to_vertex was found.
   */
  std::vector<deglib::search::ObjectDistance> hasPath(const std::vector<uint32_t>& entry_vertex_indices, const uint32_t to_vertex, const float eps, const uint32_t k) const final
  {
    const auto query = this->feature_by_index(to_vertex);
    const auto dist_func = this->feature_space_.get_dist_func();
    const auto dist_func_param = this->feature_space_.get_dist_func_param();

    // set of checked vertex ids
    auto checked_ids = std::vector<bool>(this->size());

    // items to traverse next
    auto next_vertices = deglib::search::UncheckedSet();

    // trackable information 
    auto trackback = tsl::robin_map<uint32_t, deglib::search::ObjectDistance>();

    // result set
    auto results = deglib::search::ResultSet();   

    // copy the initial entry vertices and their distances to the query into the three containers
    for (auto&& index : entry_vertex_indices) {
      checked_ids[index] = true;

      const auto feature = reinterpret_cast<const float*>(this->feature_by_index(index));
      const auto distance = dist_func(query, feature, dist_func_param);
      results.emplace(index, distance);
      next_vertices.emplace(index, distance);
      trackback.insert({index, deglib::search::ObjectDistance(index, distance)});
    }

    // search radius
    auto r = std::numeric_limits<float>::max();

    // iterate as long as good elements are in the next_vertices queue     
    auto good_neighbors = std::array<uint32_t, 256>();
    while (next_vertices.empty() == false)
    {
      // next vertex to check
      const auto next_vertex = next_vertices.top();
      next_vertices.pop();

      // max distance reached
      if (next_vertex.getDistance() > r * (1 + eps)) 
        break;

      size_t good_neighbor_count = 0;
      const auto neighbor_indices = this->neighbors_by_index(next_vertex.getInternalIndex());
      for (size_t i = 0; i < this->edges_per_vertex_; i++) {
        const auto neighbor_index = neighbor_indices[i];

        // found our target vertex, create a path back to the entry vertex
        if(neighbor_index == to_vertex) {
          auto path = std::vector<deglib::search::ObjectDistance>();
          path.emplace_back(to_vertex, 0.f);
          path.emplace_back(next_vertex.getInternalIndex(), next_vertex.getDistance());

          auto last_vertex = trackback.find(next_vertex.getInternalIndex());
          while(last_vertex != trackback.cend() && last_vertex->first != last_vertex->second.getInternalIndex()) {
            path.emplace_back(last_vertex->second.getInternalIndex(), last_vertex->second.getDistance());
            last_vertex = trackback.find(last_vertex->second.getInternalIndex());
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
             
        // check the neighborhood of this vertex later, if its good enough
        if (neighbor_distance <= r * (1 + eps)) {
          next_vertices.emplace(neighbor_index, neighbor_distance);
          trackback.insert({neighbor_index, deglib::search::ObjectDistance(next_vertex.getInternalIndex(), next_vertex.getDistance())});

          // remember the vertex, if its better than the worst in the result list
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
   * The result set contains internal indices. 
   */
  deglib::search::ResultSet search(const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count = 0) const final
  {
    if(max_distance_computation_count == 0)
      return search_func_(*this, entry_vertex_indices, query, eps, k, 0);
    else {
      const auto limited_search_func = getSearchFunction<true>(this->feature_space_);
      return limited_search_func(*this, entry_vertex_indices, query, eps, k, max_distance_computation_count);
    }
  }

  
  /**
   * The result set contains internal indices. 
   */
  template <typename COMPARATOR, bool use_max_distance_count>
  deglib::search::ResultSet searchImplExperimental(const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count) const
  {
    const auto dist_func_param = this->feature_space_.get_dist_func_param();
    uint32_t distance_computation_count = 0;

    // set of checked vertex ids
    auto checked_ids = std::vector<bool>(this->size());

    // items to traverse next
    auto next_vertices = deglib::search::UncheckedSet();
    next_vertices.reserve(k*this->edges_per_vertex_);

    // result set
    // TODO: custom priority queue with an internal Variable Length Array wrapped in a macro with linear-scan search and memcopy 
    auto results = deglib::search::ResultSet();   
    results.reserve(k);

    // copy the initial entry vertices and their distances to the query into the three containers
    for (auto&& index : entry_vertex_indices) {
      if(checked_ids[index] == false) {
        checked_ids[index] = true;

        const auto feature = reinterpret_cast<const float*>(this->feature_by_index(index));
        const auto distance = COMPARATOR::compare(query, feature, dist_func_param);
        next_vertices.emplace(index, distance);
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

    // iterate as long as good elements are in the next_vertices queue     
    auto good_neighbors = std::array<uint32_t, 256>();    // this limits the neighbor count to 256 using Variable Length Array wrapped in a macro
    auto good_neighbors_weights = std::array<float, 256>();    // this limits the neighbor count to 256 using Variable Length Array wrapped in a macro

    while (next_vertices.empty() == false)
    {
      // next vertex to check
      const auto next_vertex = next_vertices.top();
      next_vertices.pop();

      // max distance reached
      if (next_vertex.getDistance() > r * (1 + eps)) 
        break;

      size_t good_neighbor_count = 0;
      const auto neighbor_weights = this->weights_by_index(next_vertex.getInternalIndex());
      const auto neighbor_indices = this->neighbors_by_index(next_vertex.getInternalIndex());
      for (size_t i = 0; i < this->edges_per_vertex_; i++) {
        const auto neighbor_index = neighbor_indices[i];
        if (checked_ids[neighbor_index] == false)  {
          checked_ids[neighbor_index] = true;
          good_neighbors_weights[good_neighbor_count] = neighbor_weights[i];
          good_neighbors[good_neighbor_count++] = neighbor_index;
        }
      }

      if (good_neighbor_count == 0)
        continue;

      float maxDist = 0;
      float minDist = std::numeric_limits<float>::max();
      size_t minDistIndex = 0;
      MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[0])));
      for (size_t i = 0; i < good_neighbor_count; i++) {
        MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[std::min(i + 1, good_neighbor_count - 1)])));

        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_feature_vector = this->feature_by_index(neighbor_index);
        const auto neighbor_distance = COMPARATOR::compare(query, neighbor_feature_vector, dist_func_param);

        maxDist = std::max(neighbor_distance, maxDist);
             
        // check the neighborhood of this vertex later, if its good enough
        if (neighbor_distance <= r * (1 + eps)) {
          next_vertices.emplace(neighbor_index, neighbor_distance);

          if(neighbor_distance < minDist)
            minDistIndex = i;
          minDist = std::min(neighbor_distance, minDist);


          // remember the vertex, if its better than the worst in the result list
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

      // no neighbor is better than the current vertex
      // before we backtrack to an older vertices
      // we try distance estimation with the help of edge weights
      if(minDistIndex > 0 && next_vertex.getDistance() < minDist) {
        const auto neighbor_index = good_neighbors[minDistIndex];
        const auto neighbor_weight = good_neighbors_weights[minDistIndex];
        const auto neighbor_neighbor_weights = this->weights_by_index(neighbor_index);
        const auto neighbor_neighbor_indices = this->neighbors_by_index(neighbor_index);

        // check the neighbors of the good neighbors
        for (size_t i = 0; i < this->edges_per_vertex_; i++) {
          const auto neighbor_neighbor_index = neighbor_neighbor_indices[i];
          const auto neighbor_neighbor_weight = neighbor_neighbor_weights[i];

          // form a small path from current vertex, to one of its neighbors, to one of their neighbors
          // if the sum of the weights in the path is shorter than the longest edge of the current vertex
          // check the vertex at the ende of the path
          // if(checked_ids[neighbor_neighbor_index] == false) {   
          // if(checked_ids[neighbor_neighbor_index] == false && neighbor_neighbor_weight < 0) { 
          if(checked_ids[neighbor_neighbor_index] == false && neighbor_neighbor_weight <= r * (1 + eps)) { 
          // if(checked_ids[neighbor_neighbor_index] == false && neighbor_neighbor_weight > minDist) {   
          // if(checked_ids[neighbor_neighbor_index] == false && neighbor_weight + neighbor_neighbor_weight < (minDist + maxDist)/2) {              
          // if(checked_ids[neighbor_neighbor_index] == false && neighbor_neighbor_weight < maxDist) {    
          // if(checked_ids[neighbor_neighbor_index] == false && neighbor_index + neighbor_neighbor_weight < maxDist) {  
            checked_ids[neighbor_neighbor_index] = true;
            const auto neighbor_neighbor_feature_vector = this->feature_by_index(neighbor_neighbor_index);
            const auto neighbor_neighbor_distance = COMPARATOR::compare(query, neighbor_neighbor_feature_vector, dist_func_param);

            // check the neighborhood of this vertex later, if its good enough
            if (neighbor_neighbor_distance <= r * (1 + eps)) {
              next_vertices.emplace(neighbor_neighbor_index, neighbor_neighbor_distance);

              // remember the vertex, if its better than the worst in the result list
              if (neighbor_neighbor_distance < r) {
                results.emplace(neighbor_neighbor_index, neighbor_neighbor_distance);

                // update the search radius
                if (results.size() > k) {
                  results.pop();
                  r = results.top().getDistance();
                }
              }
            }
          }
        }
      }

      

      // no neighbor is better than the current vertex
      // before we backtrack to an older vertices
      // we try distance estimation with the help of edge weights
      // find potential vertices in the broder neighborhood of the current vertex which path is short regarding the weights
      // if(next_vertex.getDistance() < minDist) {

      //   // for (size_t n = 0; n < this->edges_per_vertex_; n++) {
      //   //   const auto neighbor_index = neighbor_indices[n];
      //   //   const auto neighbor_weight = neighbor_weights[n];

      //   // good_neighbors is better than all neighbors
      //   for (size_t n = 0; n < good_neighbor_count; n++) {
      //     const auto neighbor_index = good_neighbors[n];
      //     const auto neighbor_weight = good_neighbors_weights[n];
      //     const auto neighbor_neighbor_weights = this->weights_by_index(neighbor_index);
      //     const auto neighbor_neighbor_indices = this->neighbors_by_index(neighbor_index);

      //     // check the neighbors of the good neighbors
      //     for (size_t i = 0; i < this->edges_per_vertex_; i++) {
      //       const auto neighbor_neighbor_index = neighbor_neighbor_indices[i];
      //       const auto neighbor_neighbor_weight = neighbor_neighbor_weights[i];

      //       // form a small path from current vertex, to one of its neighbors, to one of their neighbors
      //       // if the sum of the weights in the path is shorter than the longest edge of the current vertex
      //       // check the vertex at the ende of the path
      //       // if(checked_ids[neighbor_neighbor_index] == false && neighbor_neighbor_weight < 0) {   
      //       if(checked_ids[neighbor_neighbor_index] == false && neighbor_neighbor_weight < minDist) {   
      //       // if(checked_ids[neighbor_neighbor_index] == false && neighbor_weight + neighbor_neighbor_weight < (minDist + maxDist)/2) {              
      //       // if(checked_ids[neighbor_neighbor_index] == false && neighbor_neighbor_weight < maxDist) {    
      //       // if(checked_ids[neighbor_neighbor_index] == false && neighbor_index + neighbor_neighbor_weight < maxDist) {   // this mistake gives even better results for SIFT1M
      //         checked_ids[neighbor_neighbor_index] = true;
      //         const auto neighbor_neighbor_feature_vector = this->feature_by_index(neighbor_neighbor_index);
      //         const auto neighbor_neighbor_distance = COMPARATOR::compare(query, neighbor_neighbor_feature_vector, dist_func_param);

      //         // check the neighborhood of this vertex later, if its good enough
      //         if (neighbor_neighbor_distance <= r * (1 + eps)) {
      //           next_vertices.emplace(neighbor_neighbor_index, neighbor_neighbor_distance);

      //           // remember the vertex, if its better than the worst in the result list
      //           if (neighbor_neighbor_distance < r) {
      //             results.emplace(neighbor_neighbor_index, neighbor_neighbor_distance);

      //             // update the search radius
      //             if (results.size() > k) {
      //               results.pop();
      //               r = results.top().getDistance();
      //             }
      //           }
      //         }
      //       }
      //     }
      //   }
      // }

      // early stop after to many computations
      if constexpr (use_max_distance_count) {
        if(distance_computation_count++ >= max_distance_computation_count)
          return results;
      }
    }

    return results;
  }

  /**
   * The result set contains internal indices. 
   */
  template <typename COMPARATOR, bool use_max_distance_count>
  deglib::search::ResultSet searchImpl(const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count) const
  {
    uint64_t hop_sum = 0;
    uint64_t dist_cal_sum = 0;
    uint64_t checked_vertices_sum = 0;


    const auto dist_func_param = this->feature_space_.get_dist_func_param();
    uint32_t distance_computation_count = 0;

    // set of checked vertex ids
    auto checked_ids = std::vector<bool>(this->size());

    // items to traverse next
    auto next_vertices = deglib::search::UncheckedSet();
    next_vertices.reserve(k*this->edges_per_vertex_);

    // result set
    // TODO: custom priority queue with an internal Variable Length Array wrapped in a macro with linear-scan search and memcopy 
    auto results = deglib::search::ResultSet();   
    results.reserve(k+1);

    // add backlog
    // auto backlog = deglib::search::ResultSet();
    // backlog.reserve(k+1);

    // copy the initial entry vertices and their distances to the query into the three containers
    for (auto&& index : entry_vertex_indices) {
      checked_vertices_sum++;
      if(checked_ids[index] == false) {
        checked_ids[index] = true;

        const auto feature = reinterpret_cast<const float*>(this->feature_by_index(index));
        const auto distance = COMPARATOR::compare(query, feature, dist_func_param);
        next_vertices.emplace(index, distance);
        results.emplace(index, distance);

        // early stop after to many computations
        dist_cal_sum++;
        if constexpr (use_max_distance_count) {
          if(distance_computation_count++ >= max_distance_computation_count)
            return results;
        }
      }
    }

    // search radius
    auto r = std::numeric_limits<float>::max();

    // iterate as long as good elements are in the next_vertices queue
    auto good_neighbors = std::array<uint32_t, 256>();    // this limits the neighbor count to 256 using Variable Length Array wrapped in a macro
    while (next_vertices.empty() == false)
    {
      // next vertex to check
      const auto next_vertex = next_vertices.top();
      next_vertices.pop();
      hop_sum++;

      // max distance reached
      if (next_vertex.getDistance() > r * (1 + eps)) 
        break;

      size_t good_neighbor_count = 0;
      const auto neighbor_indices = this->neighbors_by_index(next_vertex.getInternalIndex());
      for (size_t i = 0; i < this->edges_per_vertex_; i++) {
        const auto neighbor_index = neighbor_indices[i];
        checked_vertices_sum++;
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
             
        // check the neighborhood of this vertex later, if its good enough
        if (neighbor_distance <= r * (1 + eps)) {
            next_vertices.emplace(neighbor_index, neighbor_distance);

          // remember the vertex, if its better than the worst in the result list
          if (neighbor_distance < r) {
            results.emplace(neighbor_index, neighbor_distance);

            // update the search radius
            if (results.size() > k) {

              // update backlog
              // backlog.emplace(results.top().getInternalIndex(), results.top().getDistance());
              // if (backlog.size() > k) 
              //   backlog.pop();

              // limit result list
              results.pop();
              r = results.top().getDistance();
            }
          }
        }
      
        // early stop after to many computations
        dist_cal_sum++;
        if constexpr (use_max_distance_count) {
          if(distance_computation_count++ >= max_distance_computation_count)           
            return results;
        }
      }
    }

    results.hop_sum_ = hop_sum;
    results.dist_cal_sum_ = dist_cal_sum;
    results.checked_vertices_sum_ = checked_vertices_sum;

    // add backlog to results
    // while(backlog.size() > 0) {
    //   results.emplace(backlog.top().getInternalIndex(), backlog.top().getDistance());
    //   backlog.pop();
    // }
    return results;
  }
  // template <typename COMPARATOR, bool use_max_distance_count>
  // deglib::search::ResultSet searchImpl(const std::vector<uint32_t>& entry_vertex_indices, const std::byte* query, const float eps, const uint32_t k, const uint32_t max_distance_computation_count) const
  // {
  //   const auto dist_func_param = this->feature_space_.get_dist_func_param();
  //   uint32_t distance_computation_count = 0;

  //   // set of checked vertex ids
  //   auto checked_ids = std::vector<bool>(this->size());

  //   // items to traverse next
  //   auto next_vertices = deglib::search::UncheckedSet();
  //   next_vertices.reserve(k*this->edges_per_vertex_);

  //   // result set
  //   // TODO: custom priority queue with an internal Variable Length Array wrapped in a macro with linear-scan search and memcopy 
  //   auto results = deglib::search::ResultSet();   
  //   results.reserve(k);

  //   // copy the initial entry vertices and their distances to the query into the three containers
  //   for (auto&& index : entry_vertex_indices) {
  //     if(checked_ids[index] == false) {
  //       checked_ids[index] = true;

  //       const auto feature = reinterpret_cast<const float*>(this->feature_by_index(index));
  //       const auto distance = COMPARATOR::compare(query, feature, dist_func_param);
  //       next_vertices.emplace(index, distance);
  //       results.emplace(index, distance);

  //       // early stop after to many computations
  //       if constexpr (use_max_distance_count) {
  //         if(distance_computation_count++ >= max_distance_computation_count)
  //           return results;
  //       }
  //     }
  //   }

  //   // search radius
  //   auto r = std::numeric_limits<float>::max();

  //   // iterate as long as good elements are in the next_vertices queue
  //   auto good_neighbors = std::array<uint32_t, 256>();    // this limits the neighbor count to 256 using Variable Length Array wrapped in a macro
  //   while (next_vertices.empty() == false)
  //   {
  //     // next vertex to check
  //     const auto next_vertex = next_vertices.top();
  //     next_vertices.pop();

  //     // max distance reached
  //     if (next_vertex.getDistance() > r * (1 + eps)) 
  //       break;

  //     size_t good_neighbor_count = 0;
  //     const auto neighbor_indices = this->neighbors_by_index(next_vertex.getInternalIndex());
  //     for (size_t i = 0; i < this->edges_per_vertex_; i++) {
  //       const auto neighbor_index = neighbor_indices[i];
  //       if (checked_ids[neighbor_index] == false)  {
  //         checked_ids[neighbor_index] = true;
  //         good_neighbors[good_neighbor_count++] = neighbor_index;
  //       }
  //     }

  //     if (good_neighbor_count == 0)
  //       continue;

  //     MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[0])));
  //     for (size_t i = 0; i < good_neighbor_count; i++) {
  //       MemoryCache::prefetch(reinterpret_cast<const char*>(this->feature_by_index(good_neighbors[std::min(i + 1, good_neighbor_count - 1)])));

  //       const auto neighbor_index = good_neighbors[i];
  //       const auto neighbor_feature_vector = this->feature_by_index(neighbor_index);
  //       const auto neighbor_distance = COMPARATOR::compare(query, neighbor_feature_vector, dist_func_param);
             
  //       // check the neighborhood of this vertex later, if its good enough
  //       if (neighbor_distance <= r * (1 + eps)) {
  //           next_vertices.emplace(neighbor_index, neighbor_distance);

  //         // remember the vertex, if its better than the worst in the result list
  //         if (neighbor_distance < r) {
  //           results.emplace(neighbor_index, neighbor_distance);

  //           // update the search radius
  //           if (results.size() > k) {
  //             results.pop();
  //             r = results.top().getDistance();
  //           }
  //         }
  //       }
  //     }

  //     // early stop after to many computations
  //     if constexpr (use_max_distance_count) {
  //       if(distance_computation_count++ >= max_distance_computation_count)
  //         return results;
  //     }
  //   }

  //   return results;
  // }

  /**
   * The result set contains internal indices. 
   */
  deglib::search::ResultSet explore(const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count) const final
  {
    return explore_func_(*this, entry_vertex_index, k, max_distance_computation_count);
  }

  /**
   * The result set contains internal indices. 
   */
  template <typename COMPARATOR>
  deglib::search::ResultSet exploreImpl(const uint32_t entry_vertex_index, const uint32_t k, const uint32_t max_distance_computation_count) const
  {
    uint32_t distance_computation_count = 0;
    const auto dist_func_param = this->feature_space_.get_dist_func_param();
    const auto query = this->feature_by_index(entry_vertex_index);

    // set of checked vertex ids
    auto checked_ids = std::vector<bool>(this->size());

    // items to traverse next
    auto next_vertices = deglib::search::UncheckedSet();
    next_vertices.reserve(k*this->edges_per_vertex_);

    // result set
    auto results = deglib::search::ResultSet();   
    results.reserve(k);

    // add the entry vertex index to the vertices which gets checked next and ignore it for further checks
    {
      checked_ids[entry_vertex_index] = true;
      next_vertices.emplace(entry_vertex_index, 0);

      const auto neighbor_indices = this->neighbors_by_index(entry_vertex_index);
      const auto neighbor_weights = this->weights_by_index(entry_vertex_index);
      for (uint8_t i = 0; i < this->edges_per_vertex_; i++) {
        checked_ids[neighbor_indices[i]] = true;
        next_vertices.emplace(neighbor_indices[i], neighbor_weights[i]);
        results.emplace(neighbor_indices[i], neighbor_weights[i]);
        
        // early stop after to many computations
        if(distance_computation_count++ >= max_distance_computation_count)
          return results;
      }
    }

    // search radius
    auto r = std::numeric_limits<float>::max();

    // our eps replacement parameter
    const auto eps = std::log10(float(max_distance_computation_count)/k);

    // iterate as long as good elements are in the next_vertices queue and max_calcs is not yet reached
    auto good_neighbors = std::array<uint32_t, 256>();    // this limits the neighbor count to 256 using Variable Length Array wrapped in a macro
    while (next_vertices.empty() == false)
    {
      // next vertex to check
      const auto next_vertex = next_vertices.top();
      next_vertices.pop();

      // if no weight of this neighbor would survive the distance estimation check, stop here
      if (next_vertex.getDistance() > r * (1 + eps))
        break;

      uint8_t good_neighbor_count = 0;
      {
        const auto neighbor_indices = this->neighbors_by_index(next_vertex.getInternalIndex());
        const auto neighbor_weights = this->weights_by_index(next_vertex.getInternalIndex());
        MemoryCache::prefetch(reinterpret_cast<const char*>(neighbor_indices));
        MemoryCache::prefetch(reinterpret_cast<const char*>(neighbor_weights));
        for (uint8_t i = 0; i < this->edges_per_vertex_; i++) {
          const auto neighbor_index = neighbor_indices[i];

          if (checked_ids[neighbor_index] == false)  {
            checked_ids[neighbor_index] = true;

            // distance estimation check: allow only edges with a worst case distance < r
            // this produces slighly better results and brings the sizebound graph on par with the readonly graph when comparing speed vs quality
            if(next_vertex.getDistance() + neighbor_weights[i] < r * (1 + eps))
              good_neighbors[good_neighbor_count++] = neighbor_index;
          }
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

          // check the neighborhood of this vertex later
          next_vertices.emplace(neighbor_index, neighbor_distance);

          // remember the vertex, if its better than the worst in the result list
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
  uint8_t metric_type;
  ifstream.read(reinterpret_cast<char*>(&metric_type), sizeof(metric_type));
  uint16_t dim;
  ifstream.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  const auto feature_space = deglib::FloatSpace(dim, static_cast<deglib::Metric>(metric_type));

  // create the graph
  uint32_t size;
  ifstream.read(reinterpret_cast<char*>(&size), sizeof(size));
  uint8_t edges_per_vertex;
  ifstream.read(reinterpret_cast<char*>(&edges_per_vertex), sizeof(edges_per_vertex));

  // if no new max size is set use the size of the graph from disk
  if(new_max_size == 0) 
    new_max_size = size;
  
  // if there is a max size is should be higher than the needed graph size from disk
  if(new_max_size < size) {
    fmt::print(stderr, "The graph in the {} file has {} vertices but the new max size is {}\n", path_graph, size, new_max_size);
    perror("");
    abort();
  }

  auto graph = deglib::graph::SizeBoundedGraph(new_max_size, edges_per_vertex, std::move(feature_space), ifstream, size);
  ifstream.close();

  return graph;
}

}  // namespace deglib::graph