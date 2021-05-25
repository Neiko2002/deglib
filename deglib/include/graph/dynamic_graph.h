#pragma once

#include <cstdint>
#include <limits>
#include <math.h>

#include <fmt/core.h>
#include <tsl/robin_map.h>

#include "search.h"
#include "graph.h"
#include "repository.h"

namespace deglib::graph
{

class DynamicGraph : public deglib::search::MutableGraph {
  
  // any array of pointer to different parts in memory where the node information are stored
  // every part contains sizeof(uint16_t) nodes. The node index decribes the memory index with
  // it 16 higher bits and the index inside the memory block with its 16 lower bits.
  std::array<uint64_t, sizeof(uint16_t)> memory_ptr;

public:

  const uint8_t getEdgesPerNode() const override {
    return 0;
  }

  const deglib::SpaceInterface<float>& getFeatureSpace() const override {
    return nullptr;
  }

  inline const uint32_t getInternalIndex(const uint32_t external_label) const override {
    return 0;
  }

  inline const uint32_t getExternalLabel(const uint32_t internal_idx) const override {
    return 0;
  }

  deglib::search::ResultSet yahooSearch(const std::vector<uint32_t>& entry_node_indizies, const float* query, const float eps, const int k) const override
  {
    nullptr;
  }

  uint32_t addNode(const uint32_t external_label, const  std::byte* feature_vector, const uint32_t* neighbor_indizies, const float* neighbor_weights) override {
    return 0;
  }

  bool changeEdge(const uint32_t internal_index, const uint32_t from_neighbor_index, const uint32_t to_neighbor_index, const float to_neighbor_weight) override {
    return true;
  }

  void changeEdges(const uint32_t internal_index, const uint32_t* neighbor_indizies, const float* neighbor_weights) override {

  }

};

}  // namespace deglib::graph