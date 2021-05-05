#include <assert.h>
#include <fmt/core.h>
#include <stdio.h>
#include <tsl/robin_map.h>

#include <filesystem>
#include <fstream>
#include <iostream>

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
  std::error_code ec{};
  auto file_size = std::filesystem::file_size(fname, ec);
  if (ec != std::error_code{}) {
    fmt::print(stderr,
               "error when accessing test file, size is: {} message: {} \n",
               file_size, ec.message());
    perror("");
    abort();
  }

  auto ifstream = std::ifstream(fname, std::ios::binary);
  if (!ifstream.is_open()) {
    fmt::print(stderr, "could not open {}\n", fname);
    perror("");
    abort();
  }

  int dims;
  ifstream.read(reinterpret_cast<char*>(&dims), sizeof(int));
  assert((dims > 0 && dims < 1000000) || !"unreasonable dimension");
  assert(file_size % ((dims + 1) * 4) == 0 || !"weird file size");
  size_t n = file_size / ((dims + 1) * 4);

  *d_out = dims;
  *n_out = n;
  float* x = new float[n * (dims + 1)];
  ifstream.seekg(0);
  ifstream.read(reinterpret_cast<char*>(x), n * (dims + 1) * sizeof(float));
  if (!ifstream)
    assert(ifstream.gcount() == n * (dims + 1) || !"could not read whole file");

  // shift array to remove row headers
  for (size_t i = 0; i < n; i++)
    memmove(x + i * dims, x + 1 + i * (dims + 1), dims * sizeof(float));

  ifstream.close();
  return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int32_t* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
  return (int32_t*)fvecs_read(fname, d_out, n_out);
}

void load_test() {
  auto* path_groundtruth = "c:/Data/Feature/SIFT1M/sift_groundtruth.ivecs";
  auto* path_basedata = "c:/Data/Feature/SIFT1M/sift_base.fvecs";

  size_t dims;
  size_t count;
  float* features = fvecs_read(path_basedata, &dims, &count);
  fmt::print("dims {} count {} \n", dims, count);
}

void hashmap_test() {
}


int main() {
  fmt::print("Testing ...\n");

  size_t count = 10;
  auto dim_to_gt = tsl::robin_map<uint32_t, int32_t*>(count);
  fmt::print("Size {} \n", dim_to_gt.size());

  int32_t a[] = {4,7,9,4};
  dim_to_gt[10] = a;

  auto it = dim_to_gt.find(10);
  if (it != dim_to_gt.end()) {
    fmt::print("first {}, second {}\n", it->first, it->second);
  }

  fmt::print("Test OK\n");
  return 0;
}