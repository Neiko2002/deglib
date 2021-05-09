#include <assert.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <stdio.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>
#include  <float.h>


#include <filesystem>
#include <fstream>
#include <iostream>

#include <string>
#include <iostream>
#include <limits>

#include "arrayview.h"
#include "deglib.h"
#include "stopwatch.h"

// not very clean, but works as long as sizeof(int) == sizeof(float)
uint32_t* ivecs_read(const char* fname, size_t &d_out, size_t &n_out) {
  return (uint32_t*)deglib::fvecs_read(fname, d_out, n_out);
}

void load_sift_data_test() {
  auto* path_basedata = "c:/Data/Feature/SIFT1M/sift_base.fvecs";

  size_t dims;
  size_t count;
  float* contiguous_features = deglib::fvecs_read(path_basedata, dims, count);
  fmt::print("dims {} count {} \n", dims, count);

  // https://www.oreilly.com/library/view/understanding-and-using/9781449344535/ch04.html
  float** features = (float**)malloc(count * sizeof(float*));
  for (size_t i = 0; i < count; i++) {
    features[i] = contiguous_features + i * dims;
  }

  fmt::print("features {}\n", array_view(features[1], dims));
}

void hashmap_test() {
  size_t count = 10;
  auto dim_to_gt = tsl::robin_map<uint32_t, int32_t*>(count);
  fmt::print("Size {} \n", dim_to_gt.size());

  int32_t a[] = {4, 7, 9, 4};
  fmt::print("a {}\n", fmt::join(a, ", "));
  fmt::print("a ptr {}\n", fmt::ptr(a));

  dim_to_gt[10] = a;

  auto it = dim_to_gt.find(10);
  if (it != dim_to_gt.end()) {
    fmt::print("first {}, second {}\n", it->first, view_array(it->second, 4));
  }
}

int main() {
  fmt::print("Testing ...\n");

  StopW stopw = StopW();
  auto path_graph =
      "c:/Data/Feature/SIFT1M/"
      "k24nns_128D_L2_Path10_Rnd3+3Improve_AddK20Eps0.2_ImproveK20Eps0.025_"
      "WorstEdge0_cpp.graph";
  auto graph = deglib::load_graph(path_graph);
  float time_in_ms = stopw.getElapsedTimeMicro() / 1000;
  fmt::print("graph node count {} took {}ms\n", graph.size(), time_in_ms);

  const auto path_repository = "c:/Data/Feature/SIFT1M/sift_base.fvecs";
  auto repository = deglib::load_repository(path_repository);
  fmt::print("{} Base Features with {} dimensions \n", repository.size(), repository.dims());

  const auto path_query_repository = "c:/Data/Feature/SIFT1M/sift_query.fvecs";
  auto query_repository = deglib::load_statc_repository(path_query_repository);
  fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());

  const auto path_query_groundtruth = "c:/Data/Feature/SIFT1M/sift_groundtruth.ivecs";
  size_t dims;
  size_t count;
  auto ground_truth = ivecs_read(path_query_groundtruth, dims, count);
  fmt::print("{} ground truth {} dimensions \n", count, dims);

  const auto l2space = deglib::L2Space(repository.dims());


  // reproduceable entry point for the graph search
  auto entry_node_ids = std::vector<uint32_t>();
  entry_node_ids.reserve(1);
  auto it = repository.begin();
  if(it != repository.end())
    entry_node_ids.push_back(it.key());

  // reproduceable query for the graph search
  auto query = query_repository.getFeature(0);
  int k = 10;
  float eps = 0.1;
  auto result_queue = deglib::yahooSearch(graph, repository, entry_node_ids, query, l2space, eps, k);

  while(result_queue.size() > 0) {
    auto entry = result_queue.top();
    result_queue.pop();
    fmt::print("entry id {} and distance {} \n", entry.getId(), entry.getDistance());
  }

  fmt::print("Test OK\n");
  return 0;
}