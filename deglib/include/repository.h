#ifndef DEG_REPOSITORY_H
#define DEG_REPOSITORY_H

#include <assert.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <stdio.h>
#include <tsl/robin_map.h>

#include <filesystem>
#include <fstream>
#include <iostream>

namespace deglib {

struct StaticRepository {
  size_t dims;
  size_t count;
  float** features;
};

struct Repository {
  size_t dims;
  tsl::robin_map<uint32_t, float*> features;
};

/*****************************************************
 * I/O functions for fvecs and ivecs
 * Reference https://github.com/facebookresearch/faiss/blob/e86bf8cae1a0ecdaee1503121421ed262ecee98c/demos/demo_sift1M.cpp
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

StaticRepository load_statc_repository(const char* path_repository) {
  size_t dims;
  size_t count;
  float* contiguous_features = fvecs_read(path_repository, &dims, &count);

  // https://www.oreilly.com/library/view/understanding-and-using/9781449344535/ch04.html
  float** features = (float**)malloc(count * sizeof(float*));
  for (size_t i = 0; i < count; i++) {
    features[i] = contiguous_features + i * dims;
  }

  return {dims, count, features};
}

Repository load_repository(const char* path_repository) {
  size_t dims;
  size_t count;
  float* contiguous_features = fvecs_read(path_repository, &dims, &count);

  auto feature_map = tsl::robin_map<uint32_t, float*>(count);
  for (size_t i = 0; i < count; i++) {
    feature_map[i] = contiguous_features + i * dims;
  }

  return {dims, feature_map};
}

}  // namespace deglib

#endif