#include <assert.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <stdio.h>
#include <tsl/robin_map.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "stopwatch.h"
#include "arrayview.h"
#include "deglib.h"

// not very clean, but works as long as sizeof(int) == sizeof(float)
int32_t* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
  return (int32_t*)deglib::fvecs_read(fname, d_out, n_out);
}

void load_sift_data_test() {
  auto* path_basedata = "c:/Data/Feature/SIFT1M/sift_base.fvecs";

  size_t dims;
  size_t count;
  float* contiguous_features = deglib::fvecs_read(path_basedata, &dims, &count);
  fmt::print("dims {} count {} \n", dims, count);

  // https://www.oreilly.com/library/view/understanding-and-using/9781449344535/ch04.html
  float** features = (float **) malloc(count * sizeof(float *));
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


void load_graph_test() {
  auto path_graph =
      "c:/Data/Feature/SIFT1M/"
      "k24nns_128D_L2_Path10_Rnd3+3Improve_AddK20Eps0.2_ImproveK20Eps0.025_"
      "WorstEdge0_cpp.graph";

  StopW stopw = StopW();
  auto graph = deglib::load_graph(path_graph);
  float time_in_ms = stopw.getElapsedTimeMicro() / 1000;
  fmt::print("graph node count {} took {}ms\n", graph.size(), time_in_ms);

  auto it = graph.begin();
  fmt::print("node id {}\n", it.value());
}



int main() {
  fmt::print("Testing ...\n");


  auto* path_repository = "c:/Data/Feature/SIFT1M/sift_base.fvecs";
  auto repo = deglib::load_repository(path_repository);
  fmt::print("repo dims {}\n", repo.dims);

  fmt::print("test_func {}\n", deglib::test_func());


  fmt::print("Test OK\n");
  return 0;
}