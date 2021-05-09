#pragma once

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

/**
 * A repository of float feature vectors.
 */
class FeatureRepository {
  virtual const float* getFeature(const uint32_t nodeid) const = 0;
};

/**
 * A repository of float feature vectors. Since the repository deals
 * with static data, a single contiguous array is preserved internally.
 */ 
class StaticFeatureRepository : public FeatureRepository {
 public:
  StaticFeatureRepository(const float* contiguous_features, const size_t dims,
                          const size_t count)
      : contiguous_features_{contiguous_features}, dims_{dims}, count_{count} {}

  const size_t& dims() const { return dims_; }
  const size_t& size() const { return count_; }

  const float* getFeature(const uint32_t nodeid) const override {
    return contiguous_features_ + nodeid * dims_;
  }

 private:
  const size_t dims_;
  const size_t count_;
  const float* contiguous_features_;
};

/**
 * A repository of float feature vectors. This  the repository deals
 * with static data, a single contiguous array is preserved internally.
 */ 
class DynamicFeatureRepository : public FeatureRepository {
 public:
  DynamicFeatureRepository(const tsl::robin_map<uint32_t, const float*> features,
                           const size_t dims)
      : features_{features}, dims_{dims} {}

  const size_t& dims() const { return dims_; }
  const size_t size() const { return features_.size(); }

  const float* getFeature(const uint32_t nodeid) const override {
    return features_.at(nodeid);
  }

  tsl::robin_map<uint32_t, const float*>::iterator begin() {
    return features_.begin();
  }

  tsl::robin_map<uint32_t, const float*>::iterator end() {
    return features_.end();
  }

  tsl::robin_map<uint32_t, const float*>::const_iterator cbegin() const {
    return features_.cbegin();
  }

  tsl::robin_map<uint32_t, const float*>::const_iterator cend() const {
    return features_.cend();
  }

  void addFeature(const uint32_t nodeid, const float* feature) {
    features_[nodeid] = feature;
  }

  void deleteFeature(const uint32_t nodeid) {
    features_.erase(nodeid);
  }

 private:
  const size_t dims_;
  tsl::robin_map<uint32_t, const float*> features_;
};

/*****************************************************
 * I/O functions for fvecs and ivecs
 * Reference
 *https://github.com/facebookresearch/faiss/blob/e86bf8cae1a0ecdaee1503121421ed262ecee98c/demos/demo_sift1M.cpp
 *****************************************************/

float* fvecs_read(const char* fname, size_t &d_out, size_t &n_out) {
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

  d_out = dims;
  n_out = n;

  // TODO use make_unique
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

StaticFeatureRepository load_statc_repository(const char* path_repository) {
  size_t dims;
  size_t count;
  const auto contiguous_features = fvecs_read(path_repository, dims, count);

  // https://www.oreilly.com/library/view/understanding-and-using/9781449344535/ch04.html
  // float** features = (float**)malloc(count * sizeof(float*));
  // for (size_t i = 0; i < count; i++) {
  //  features[i] = contiguous_features + i * dims;
  //}

  return StaticFeatureRepository(contiguous_features, dims, count);
}

DynamicFeatureRepository load_repository(const char* path_repository) {
  size_t dims;
  size_t count;
  const auto contiguous_features = fvecs_read(path_repository, dims, count);

  // TODO use shared_ptr in the map 
  auto feature_map = tsl::robin_map<uint32_t, const float*>(count);
  for (size_t i = 0; i < count; i++) {
    feature_map[i] = contiguous_features + i * dims;
  }

  return DynamicFeatureRepository(feature_map, dims);
}

}  // namespace deglib
