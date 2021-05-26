#pragma once

#include <cstdint>
#include <limits>
#include <math.h>

#include <fmt/core.h>
#include <tsl/robin_map.h>

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

  using SEARCHFUNC = deglib::search::ResultSet (*)(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const float* query, const float eps, const int k);

  inline static deglib::search::ResultSet searchL2Ext16(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const float* query, const float eps, const int k) {
    return graph.yahooSearchImpl<deglib::distances::L2Float16Ext>(entry_node_indizies, query, eps, k);
  }

  inline static deglib::search::ResultSet searchL2Ext4(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const float* query, const float eps, const int k) {
    return graph.yahooSearchImpl<deglib::distances::L2Float4Ext>(entry_node_indizies, query, eps, k);
  }

  inline static deglib::search::ResultSet searchL2Ext16Residual(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const float* query, const float eps, const int k) {
    return graph.yahooSearchImpl<deglib::distances::L2Float16ExtResiduals>(entry_node_indizies, query, eps, k);
  }

  inline static deglib::search::ResultSet searchL2Ext4Residual(const ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const float* query, const float eps, const int k) {
    return graph.yahooSearchImpl<deglib::distances::L2Float4ExtResiduals>(entry_node_indizies, query, eps, k);
  }

  static SEARCHFUNC getSearchFunction(const size_t dim) {
    if (dim % 16 == 0)
      return deglib::graph::ReadOnlyGraph::searchL2Ext16;
    else if (dim % 4 == 0)
      return deglib::graph::ReadOnlyGraph::searchL2Ext4;
    else if (dim > 16)
      return deglib::graph::ReadOnlyGraph::searchL2Ext16Residual;
    else if (dim > 4)
      return deglib::graph::ReadOnlyGraph::searchL2Ext4Residual;
    else
      return deglib::graph::ReadOnlyGraph::searchL2Ext16;
  }
  
  static uint32_t compute_aligned_byte_size_per_node(const uint8_t edges_per_node, const uint16_t feature_byte_size) {
    if constexpr(alignment == 0)
      return  uint32_t(feature_byte_size) + uint32_t(edges_per_node) * sizeof(uint32_t) + sizeof(uint32_t);
    else {
      const uint32_t byte_size = uint32_t(feature_byte_size) + uint32_t(edges_per_node) * sizeof(uint32_t) + sizeof(uint32_t);
      return ((byte_size + alignment - 1) / alignment) * alignment;
    }
  }

  static std::byte* compute_aligned_pointer(const std::unique_ptr<std::byte[]>& arr) {
    if constexpr(alignment == 0)
      return arr.get();
    else {
      auto unaliged_address = (uint64_t) arr.get();
      auto aligned_address = ((unaliged_address + alignment - 1) / alignment) * alignment;
      auto address_alignment = aligned_address - unaliged_address;
      return arr.get() + address_alignment;
    }
  }

  static const uint32_t alignment = 32; // alignment of node information in bytes

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

    // distance calculation function between feature vectors of two graph nodes
  const deglib::L2Space feature_space_;

 public:
  ReadOnlyGraph(const uint32_t max_node_count, const uint8_t edges_per_node, const deglib::L2Space feature_space)
      : edges_per_node_(edges_per_node), max_node_count_(max_node_count), feature_byte_size_(uint16_t(feature_space.get_data_size())), 
        byte_size_per_node_(compute_aligned_byte_size_per_node(edges_per_node, uint16_t(feature_space.get_data_size()))), 
        neighbor_indizies_offset_(uint32_t(feature_space.get_data_size())), feature_space_(feature_space),
        external_label_offset_(uint32_t(feature_space.get_data_size()) + uint32_t(edges_per_node) * sizeof(uint32_t)), search_func_(getSearchFunction(feature_space.dim())),
        nodes_(std::make_unique<std::byte[]>(max_node_count * byte_size_per_node_ + alignment)), nodes_memory_(compute_aligned_pointer(nodes_)), label_to_index_(max_node_count) {
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
  const size_t size() const override {
    return label_to_index_.size();
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
  inline auto getNode(const uint32_t internal_idx) const {
    return nodes_memory_ + internal_idx * byte_size_per_node_;
  }

  inline auto getFeatureVector(const uint32_t internal_idx) const {
    return getNode(internal_idx);
  }

 

public:
  /**
   * convert an external label to an internal index
   */ 
  inline const uint32_t getInternalIndex(const uint32_t external_label) const override {
    return label_to_index_.find(external_label)->second;
  }

  inline const uint32_t getExternalLabel(const uint32_t internal_idx) const override {
    return *reinterpret_cast<const int32_t*>(getNode(internal_idx) + external_label_offset_);
  }

  inline const uint32_t* getNeighborIndizies(const uint32_t internal_idx) const override {
    return reinterpret_cast<uint32_t*>(getNode(internal_idx) + neighbor_indizies_offset_);
  }

  /**
   * Add a new node. The neighbor indizies and feature vectors will be copied.
   */
  void addNode(const uint32_t external_label, const float* feature_vector, const uint32_t* neighbor_indizies) {
    const auto new_internal_index = static_cast<uint32_t>(label_to_index_.size());
    label_to_index_.emplace(external_label, new_internal_index);

    auto node_memory = getNode(new_internal_index);
    std::memcpy(node_memory, feature_vector, feature_byte_size_);
    std::memcpy(node_memory + neighbor_indizies_offset_, neighbor_indizies, uint32_t(edges_per_node_) * 4);
    std::memcpy(node_memory + external_label_offset_, &external_label, 4);
  }

  /**
   * The result set contains internal indizies. 
   */
  deglib::search::ResultSet yahooSearch(const std::vector<uint32_t>& entry_node_indizies, const float* query, const float eps, const int k) const override
  {
    return search_func_(*this, entry_node_indizies, query, eps, k);
  }

  /**
   * The result set contains internal indizies. 
   */
  template <typename COMPARATOR>
  deglib::search::ResultSet yahooSearchImpl(const std::vector<uint32_t>& entry_node_indizies, const float* query, const float eps, const int k) const
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

      const auto feature = reinterpret_cast<const float*>(this->getFeatureVector(index));
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
      const auto neighbor_indizies = this->getNeighborIndizies(next_node.getId());
      for (size_t i = 0; i < this->edges_per_node_; i++) {
        const auto neighbor_index = neighbor_indizies[i];
        if (checked_ids[neighbor_index] == false)  {
          checked_ids[neighbor_index] = true;
          good_neighbors[good_neighbor_count++] = neighbor_index;
        }
      }

      if (good_neighbor_count == 0)
        continue;

      MemoryCache::prefetch(reinterpret_cast<const char*>(this->getFeatureVector(good_neighbors[0])));
      for (size_t i = 0; i < good_neighbor_count; i++) {
        MemoryCache::prefetch(reinterpret_cast<const char*>(this->getFeatureVector(good_neighbors[std::min(i + 1, good_neighbor_count - 1)])));

        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_feature_vector = this->getFeatureVector(neighbor_index);
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
 **/
auto load_readonly_graph(const char* path_graph, const deglib::FeatureRepository &repository)
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

  // read the entire file into a buffer
  auto buffer = std::make_unique<char[]>(file_size);
  if (!ifstream.read(buffer.get(), file_size))
  {
    fmt::print(stderr, "unable to read file content {}\n", path_graph);
    perror("");
    abort();
  }
  ifstream.close();

  // the file only contains ints and floats
  auto file_values = (uint32_t*)buffer.get();
  const uint32_t node_count = *(file_values + 0);
  const auto edges_per_node = static_cast<uint8_t>(*(file_values + 2));         // expecting no more than 256 edges per node
  const auto distance_space = deglib::L2Space(repository.dims());
  auto graph = deglib::graph::ReadOnlyGraph(node_count, edges_per_node, std::move(distance_space));

  // This dataset has a natural order, SIFT feature with similar indizies are more similar to each other
  auto node_order = std::vector<std::pair<uint32_t, uint32_t>>();
  node_order.reserve(node_count);
  for (uint32_t node_idx = 0; node_idx < node_count; node_idx++) {
     const auto node_id = *(file_values + node_idx * (edges_per_node*2+2) + 1);
     node_order.emplace_back(node_id, node_idx);
  }
  std::sort(node_order.begin(), node_order.end(), [](const auto& x, const auto& y){return x.first < y.first;});

  auto neighbor_ids = std::vector<uint32_t>(edges_per_node);
  auto neighbor_weights = std::vector<std::pair<uint32_t, float>>();
  for (auto &&pair : node_order) {
    auto node = file_values + pair.second * (edges_per_node*2+2) + 3;

    // Iterating over near neighbors first, saturates the radius in the search algorithms faster with a small value
    neighbor_weights.clear();
    for (uint32_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {
      auto neighbor_id = *(node++);
      auto neighbor_weight = *reinterpret_cast<float*>(node++);
      neighbor_weights.emplace_back(neighbor_id, neighbor_weight);
    } 
    std::sort(neighbor_weights.begin(), neighbor_weights.end(), [](const auto& x, const auto& y){return x.second < y.second;});

    neighbor_ids.clear();
    for (auto &&neighbor : neighbor_weights) {
      neighbor_ids.emplace_back(neighbor.first);
    }

    graph.addNode(pair.first, repository.getFeature(pair.first), neighbor_ids.data());
  }

  return graph;
}



}  // namespace deglib::graph