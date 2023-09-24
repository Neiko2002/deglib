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
  
  // any array of pointer to different parts in memory where the vertex information are stored
  // every part contains sizeof(uint16_t) vertices. The vertex index decribes the memory index with
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

  inline const std::byte* getFeatureVector(const uint32_t internal_idx) const override {
    return nullptr;
  }

  inline const uint32_t* getNeighborIndices(const uint32_t internal_idx) const override {
    return nullptr;
  }

  inline const float* getNeighborWeights(const uint32_t internal_idx) const override {
    return nullptr;
  }

  inline const float getEdgeWeight(const uint32_t from_neighbor_index, const uint32_t to_neighbor_index) const override {
    return 0;
  }

  inline const bool hasNode(const uint32_t external_label) const override {
    return false;
  }

  inline const bool hasEdge(const uint32_t internal_index, const uint32_t neighbor_index) const override {
    return false;
  }

  const bool saveGraph(const char* path_to_graph) const override {
    return true;
  }

  std::vector<deglib::search::ObjectDistance> hasPath(const std::vector<uint32_t>& entry_vertex_indices, const uint32_t to_vertex, const float eps, const int k) const override {

    return std::vector<deglib::search::ObjectDistance>();
  }

  deglib::search::ResultSet search(const std::vector<uint32_t>& entry_vertex_indices, const float* query, const float eps, const int k) const override
  {
    nullptr;
  }

  uint32_t addNode(const uint32_t external_label, const  std::byte* feature_vector) override {
    return 0;
  }

  std::vector<uint32_t> removeNode(const uint32_t external_labelr) override {
    return std::vector<uint32_t>(0);
  }

  bool changeEdge(const uint32_t internal_index, const uint32_t from_neighbor_index, const uint32_t to_neighbor_index, const float to_neighbor_weight) override {
    return true;
  }

  void changeEdges(const uint32_t internal_index, const uint32_t* neighbor_indices, const float* neighbor_weights) override {

  }

};

}  // namespace deglib::graph