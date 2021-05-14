#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_set>

#include <fmt/core.h>
#include <tsl/robin_hash.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "deglib.h"

namespace deglib
{

/**
 * A size bounded undirected n-regular graph.
 * 
 * The node count and number of edges per nodes is bounded at construction time.
 * Therefore the graph is n-regular where n is the number of eddes per node.
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
class ReadOnlyGraph : public SearchGraph {
  const uint32_t max_node_count_;
  const uint8_t edges_per_node_;
  const uint16_t feature_byte_size_;

  const uint32_t byte_size_per_node_;
  const uint32_t neighbor_indizies_offset_;
  const uint32_t external_label_offset_;

  // list of nodes (node: feature vector, indizies of neighbor nodes, external label)
  std::unique_ptr<char[]> nodes_;

  // map from the label of a node to the internal node index
  tsl::robin_map<uint32_t, uint32_t> label_to_index_;

  // distance calculation function between feature vectors of two graph nodes
  deglib::L2Space distance_space_;

 public:
  ReadOnlyGraph(const uint32_t max_node_count, const uint8_t edges_per_node, const uint16_t feature_byte_size, deglib::L2Space distance_space)
      : edges_per_node_(edges_per_node), max_node_count_(max_node_count), feature_byte_size_(feature_byte_size), 
        byte_size_per_node_(uint32_t(feature_byte_size) + uint32_t(edges_per_node) * 4 + 4), 
        neighbor_indizies_offset_(uint32_t(feature_byte_size)), distance_space_(distance_space),
        external_label_offset_(uint32_t(feature_byte_size) + uint32_t(edges_per_node) * 4),
        nodes_(std::make_unique<char[]>(byte_size_per_node_ * max_node_count)), label_to_index_(max_node_count) {
  }

  /**
   * Takes ownership of the nodes unique_ptr from the other object
   * 
   * Temporarly removes the const-ness of the unique_ptr.
   * https://stackoverflow.com/questions/29194304/move-constructor-involving-const-unique-ptr#comment46602858_29194685
   */
  /*
  ReadOnlyGraph(const SizeBoundedGraph&& other) 
      : max_node_count_(other.max_node_count_), edges_per_node_(other.edges_per_node_), feature_byte_size_(other.feature_byte_size_),
        byte_size_per_node_(other.byte_size_per_node_), neighbor_indizies_offset_(other.neighbor_indizies_offset_), 
        external_label_offset_(other.external_label_offset_), nodes_(std::move(const_cast<std::unique_ptr<char[]>&>(other.nodes_))), 
        label_to_index_(std::move(other.label_to_index_)) {
  }
  */

  /**
   * Maximal capacity of nodes of the graph
   */ 
  const auto max_size() const {
    return max_node_count_;
  }

  /**
   * Number of nodes in th graph
   */
  const auto size() const {
    return label_to_index_.size();
  }

  /**
   * Number of edges per node 
   */
  const auto edges_per_node() const {
    return edges_per_node_;
  }

  /**
   * hash map to convert from external label to internal index
   */ 
  const auto label_to_index() const {
    return label_to_index_;
  }

  /**
   * convert an external label to an internal index
   */ 
  const uint32_t getInternalIndex(const uint32_t external_label) const {
    return label_to_index_.find(external_label)->second;
  }
  
  inline auto getNode(const uint32_t internal_idx) const {
    return nodes_.get() + internal_idx * byte_size_per_node_;
  }

  inline auto getNode(const int32_t external_label) const {
    return getNode(getInternalIndex(external_label));
  }

  inline auto getFeatureVector(const char* node) const {
    return node;
  }

  inline auto getFeatureVector(const uint32_t internal_idx) const {
    return nodes_.get() + internal_idx * byte_size_per_node_;
  }

  inline auto getNeighborIndizies(const char* node) const {
    return reinterpret_cast<const uint32_t*>(node + neighbor_indizies_offset_);
  }

  inline auto getNeighborIndizies(const uint32_t internal_idx) const {
    return reinterpret_cast<uint32_t*>(nodes_.get() + internal_idx * byte_size_per_node_ + neighbor_indizies_offset_);
  }

  inline auto getExternalLabel(const char* node) const {
    return *reinterpret_cast<const int32_t*>(node + external_label_offset_);
  }

  inline auto getExternalLabel(const uint32_t internal_idx) const {
    return *reinterpret_cast<const int32_t*>(nodes_.get() + internal_idx * byte_size_per_node_ + external_label_offset_);
  }

  /**
   * Add a new node. The neighbor indizies can be 
   */
  void addNode(const uint32_t external_label, const char* feature_vector, const uint32_t* neighbor_indizies) {
    const auto new_internal_index = static_cast<uint32_t>(label_to_index_.size());
    label_to_index_.emplace(external_label, new_internal_index);

    const auto node_memory = nodes_.get() + new_internal_index * byte_size_per_node_;
    std::memcpy(node_memory, feature_vector, feature_byte_size_);
    std::memcpy(node_memory + neighbor_indizies_offset_, neighbor_indizies, uint32_t(edges_per_node_) * 4);
    std::memcpy(node_memory + external_label_offset_, &external_label, 4);
  }

  /**
   * The result set contains internal indizies. 
   */
  deglib::ResultSet yahooSearch(const std::vector<uint32_t>& entry_node_indizies, const float* query, const float eps, const int k) const override
  {
    const auto dist_func = this->distance_space_.get_dist_func();
    const auto dist_func_param = this->distance_space_.get_dist_func_param();

    auto entry_nodes = std::vector<deglib::ObjectDistance>();
    entry_nodes.reserve(entry_node_indizies.size());
    for (auto&& index : entry_node_indizies)
    {
      const auto feature = reinterpret_cast<const float*>(this->getFeatureVector(index));
      const auto distance = dist_func(query, feature, dist_func_param);
      entry_nodes.emplace_back(index, distance);
    }
    
    return yahooSearch(entry_nodes, query, eps, k);
  }

  /**
   * The result set contains internal indizies. 
   */
  deglib::ResultSet yahooSearch(const std::vector<deglib::ObjectDistance>& entry_nodes, const float* query, const float eps, const int k) const
  {
    const auto dist_func = this->distance_space_.get_dist_func();
    const auto dist_func_param = this->distance_space_.get_dist_func_param();

    // set of checked node ids
    auto checked_ids = std::vector<bool>(this->size());
    for (auto& node : entry_nodes) checked_ids[node.getId()] = true;

    // items to traverse next, start with the initial entry nodes
    auto internal_next_nodes_vector = std::vector<deglib::ObjectDistance>();
    internal_next_nodes_vector.reserve(1000);
    auto next_nodes = deglib::UncheckedSet(std::greater<deglib::ObjectDistance>(), std::move(internal_next_nodes_vector));
    for (auto& node : entry_nodes) next_nodes.push(node);

    // result set
    auto internal_result_vector = std::vector<deglib::ObjectDistance>();
    internal_result_vector.reserve(k + 1);
    auto results = deglib::ResultSet(std::less<deglib::ObjectDistance>(), std::move(internal_result_vector));
    for (auto& node : entry_nodes) results.push(node);

    // search radius
    auto r = std::numeric_limits<float>::max();

    // iterate as long as good elements are in the next_nodes queue     
    size_t good_neighbor_count = 0;
    auto good_neighbors = std::array<uint32_t, 100>();    // this limits the neighbor count to 100
    while (next_nodes.empty() == false)
    {
      // next node to check
      const auto next_node = next_nodes.top();
      next_nodes.pop();

      // max distance reached
      if (next_node.getDistance() > r * (1 + eps)) 
        break;

      good_neighbor_count = 0;
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

      _mm_prefetch(this->getFeatureVector(good_neighbors[0]), _MM_HINT_T0); 
      const auto good_neighbor_size = good_neighbor_count;
      for (size_t i = 0; i < good_neighbor_size; i++) {
        _mm_prefetch(this->getFeatureVector(good_neighbors[std::min(i + 1, good_neighbor_size - 1)]), _MM_HINT_T0); 

        const auto neighbor_index = good_neighbors[i];
        const auto neighbor_feature_vector = this->getFeatureVector(neighbor_index);
        const auto neighbor_distance = dist_func(query, neighbor_feature_vector, dist_func_param);

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
  const uint32_t node_count = *(file_values++);
  const auto edges_per_node = static_cast<uint8_t>(*(file_values + 1));         // expecting no more than 256 edges per node
  const auto feature_byte_size = static_cast<uint16_t>(repository.dims() * 4);  // expecting no more than 4096 float values
  const auto distance_space = deglib::L2Space(repository.dims());
  auto graph = deglib::ReadOnlyGraph(node_count, edges_per_node, feature_byte_size, distance_space);

  // The node and neighbor ids from the file represent external label.
  // These label will later be replaced with the internal indizies.
  auto neighbor_ids = std::vector<uint32_t>(edges_per_node);
  for (uint32_t node_idx = 0; node_idx < node_count; node_idx++) {
    const auto node_id = *(file_values++);
    /*edge count*/ file_values++;

    neighbor_ids.clear();
    for (uint32_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {
      neighbor_ids.emplace_back(*(file_values++));
      /*edge weight*/ file_values++;
    }  

    graph.addNode(node_id, reinterpret_cast<const char*>(repository.getFeature(node_id)), neighbor_ids.data());
  }
  
  // Replace the external label of the neighbor list with the internal indizies.
  for (uint32_t node_idx = 0; node_idx < node_count; node_idx++) {
    auto neighbors = graph.getNeighborIndizies(node_idx);
    for (uint32_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {
      neighbors[edge_idx] = graph.getInternalIndex(neighbors[edge_idx]);
    }
  }

  return graph;
}



}  // namespace deglib