#include <cstdint>
#include <limits>
#include <queue>

#include <fmt/core.h>
#include <tsl/robin_hash.h>
#include <tsl/robin_map.h>

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

  // list of nodes (node: feature vector, indizies of neighbor nodes, external label)
  tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, const float*>> nodes_;



  // distance calculation function between feature vectors of two graph nodes
  const deglib::L2Space distance_space_;

 public:
  ReadOnlyGraph(const uint32_t max_node_count, const deglib::L2Space distance_space)
      : max_node_count_(max_node_count), distance_space_(distance_space), nodes_{max_node_count} {
  }

  /**
   * Number of nodes in th graph
   */
  const auto size() const {
    return nodes_.size();
  }

  auto& nodes() {
    return nodes_;
  }

  /**
   * Add a new node. The neighbor indizies can be 
   */
  void addNode(const uint32_t id, const tsl::robin_map<uint32_t, const float*>& edges) {
    nodes_[id] = std::move(edges);
  }

  /**
   * The result set contains internal indizies. 
   */
  deglib::ResultSet yahooSearch(const std::vector<uint32_t>& entry_node_id, const float* query, const float eps, const int k) const override
  {
    const auto dist_func = distance_space_.get_dist_func();
    const auto dist_func_param = distance_space_.get_dist_func_param();

    auto entry_nodes = std::vector<deglib::ObjectDistance>();
    entry_nodes.reserve(entry_node_id.size());
    for (auto&& id : entry_node_id)
    {
      const auto& node = nodes_.at(id);
      const auto neighbor_id = node.begin().key();
      const auto& neighbor = nodes_.at(neighbor_id); // we need a neighbor to get the feature of id
      const auto feature_vector = neighbor.at(id);
      const auto distance = dist_func(query, feature_vector, dist_func_param);
      entry_nodes.emplace_back(id, distance);
    }
    
    return yahooSearch(entry_nodes, query, eps, k);
  }

  /**
   * The result set contains internal indizies. 
   */
  deglib::ResultSet yahooSearch(const std::vector<deglib::ObjectDistance>& entry_nodes, const float* query, const float eps, const int k) const
  {
    const auto dist_func = distance_space_.get_dist_func();
    const auto dist_func_param = distance_space_.get_dist_func_param();

    // set of checked node ids
    auto checked_ids = std::vector<bool>(nodes_.size());
    for (auto& node : entry_nodes) checked_ids[node.getId()] = true;

    // items to traverse next, start with the initial entry nodes
    auto next_nodes = deglib::UncheckedSet();
    for (auto& node : entry_nodes) next_nodes.push(node);

    // result set
    auto results = deglib::ResultSet();   // custom priority queue with an internal Variable Length Array wrapped in a macro with scan search and memcopy 
    for (auto& node : entry_nodes) results.push(node);

    // search radius
    auto r = std::numeric_limits<float>::max();

    // iterate as long as good elements are in the next_nodes queue     
    while (next_nodes.empty() == false)
    {
      // next node to check
      const auto next_node = next_nodes.top();
      next_nodes.pop();

      // max distance reached
      if (next_node.getDistance() > r * (1 + eps)) 
        break;

      const auto& edges = nodes_.at(next_node.getId());
      for (auto&& neighbor : edges) {
        const auto neighbor_id = neighbor.first;
        if (checked_ids[neighbor_id] == false)  {
          checked_ids[neighbor_id] = true;

          const auto neighbor_distance = dist_func(query, neighbor.second, dist_func_param);
              
          // check the neighborhood of this node later, if its good enough
          if (neighbor_distance <= r * (1 + eps)) {
              next_nodes.emplace(neighbor_id, neighbor_distance);

            // remember the node, if its better than the worst in the result list
            if (neighbor_distance < r) {
              results.emplace(neighbor_id, neighbor_distance);

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

    return results;
  }
};


/**
 * Load the graph
 **/
auto load_readonly_graph(const char* path_graph, const deglib::FeatureRepository& repository)
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
  const auto distance_space = deglib::L2Space(repository.dims());
  auto graph = deglib::ReadOnlyGraph(node_count, distance_space);

  // This dataset has a natural order, SIFT feature with similar indizies are more similar to each other
  for (uint32_t node_idx = 0; node_idx < node_count; node_idx++) {
    const auto node_id = *(file_values++);
    const auto edge_count = *(file_values++);
    
    // Iterating over near neighbors first, saturates the radius in the search algorithms faster with a small value
    auto neighbor_weights = std::vector<std::pair<uint32_t, float>>();
    for (uint32_t edge_idx = 0; edge_idx < edge_count; edge_idx++) {
      auto neighbor_id = *(file_values++);
      auto neighbor_weight = *reinterpret_cast<float*>(file_values++);
      neighbor_weights.emplace_back(neighbor_id, neighbor_weight);
    } 
    std::sort(neighbor_weights.begin(), neighbor_weights.end(), [](const auto& x, const auto& y){return x.second < y.second;});

    auto edges = tsl::robin_map<uint32_t, const float*>(edge_count);
    for (auto &&neighbor : neighbor_weights) 
      edges[neighbor.first] = repository.getFeature(neighbor.first);
    
    graph.addNode(node_id, std::move(edges));
  }

  return std::move(graph);
}



}  // namespace deglib