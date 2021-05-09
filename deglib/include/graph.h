#pragma once

#include <assert.h>
#include <tsl/robin_map.h>

#include <filesystem>
#include <fstream>
#include <iostream>

namespace deglib {


class Graph {
 public:
  Graph(tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, float>> nodes) : nodes_{nodes} {}

  const size_t size() const { return nodes_.size(); }

  const tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, float>>& nodes() const { return nodes_; }

  const tsl::robin_map<uint32_t, float>& edges(const uint32_t nodeid) const { return nodes_.at(nodeid); }

 private:
  tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, float>> nodes_;
};

/**
 * Load the graph
 **/
Graph load_graph(
    const char* path_graph) {
  std::error_code ec{};
  auto file_size = std::filesystem::file_size(path_graph, ec);
  if (ec != std::error_code{}) {
    fmt::print(stderr,
               "error when accessing test file, size is: {} message: {} \n",
               file_size, ec.message());
    perror("");
    abort();
  }

  auto ifstream = std::ifstream(path_graph, std::ios::binary);
  if (!ifstream.is_open()) {
    fmt::print(stderr, "could not open {}\n", path_graph);
    perror("");
    abort();
  }

  // read the entire file into a buffer
  auto buffer = std::make_unique<char[]>(file_size);
  if (!ifstream.read(buffer.get(), file_size)) {
    fmt::print(stderr, "unable to read file content {}\n", path_graph);
    perror("");
    abort();
  }

  // the file only contains ints and floats
  auto file_values = (uint32_t*)buffer.get();
  const uint32_t nodeCount = *(file_values++);
  auto nodes = tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, float>>(nodeCount);
  for (size_t nodeIdx = 0; nodeIdx < nodeCount; nodeIdx++) {
    const uint32_t nodeId = *(file_values++);
    const uint32_t edgeCount = *(file_values++);

    auto edges = tsl::robin_map<uint32_t, float>(edgeCount);
    for (size_t edgeIdx = 0; edgeIdx < edgeCount; edgeIdx++) {
      const uint32_t neighborId = *(file_values++);
      const float weight = *(float*)(file_values++);

      edges[neighborId] = weight;
    }

    nodes[nodeId] = edges;
  }

  /*
    // read the file step by step
    uint32_t nodeCount;
    ifstream.read(reinterpret_cast<char*>(&nodeCount), sizeof(uint32_t));
    fmt::print("node count {} \n", nodeCount);

    auto graph =
        tsl::robin_map<uint32_t, tsl::robin_map<uint32_t, float>>(nodeCount);

    for (size_t nodeIdx = 0; nodeIdx < nodeCount; nodeIdx++) {
      uint32_t nodeId;
      ifstream.read(reinterpret_cast<char*>(&nodeId), sizeof(uint32_t));

      uint32_t edgeCount;
      ifstream.read(reinterpret_cast<char*>(&edgeCount), sizeof(uint32_t));

      auto edges = tsl::robin_map<uint32_t, float>(edgeCount);
      for (size_t edgeIdx = 0; edgeIdx < edgeCount; edgeIdx++) {
        uint32_t neighborId;
        ifstream.read(reinterpret_cast<char*>(&neighborId), sizeof(int));

        float weight;
        ifstream.read(reinterpret_cast<char*>(&weight), sizeof(float));

        edges[neighborId] = weight;
      }

      graph[nodeId] = edges;
    }
  */

  ifstream.close();
  return Graph(nodes);
}
    
}  // namespace deglib
