#pragma once

#include <filesystem>
#include <fstream>

#include <assert.h>
#include <stdio.h>

/*****************************************************
 * I/O functions for fvecs and ivecs
 * Reference
 *https://github.com/facebookresearch/faiss/blob/e86bf8cae1a0ecdaee1503121421ed262ecee98c/demos/demo_sift1M.cpp
 *****************************************************/


inline bool exists_test(const std::string &name)
{
    auto f = std::ifstream(name.c_str());
    return f.good();
}

float* fvecs_read(const char* fname, size_t &d_out, size_t &n_out) {
  std::error_code ec{};
  auto file_size = std::filesystem::file_size(fname, ec);
  if (ec != std::error_code{}) {
    std::fprintf(stderr, "error when accessing test file, size is: %zd message: %s \n", file_size, ec.message().c_str());
    perror("");
    abort();
  }

  auto ifstream = std::ifstream(fname, std::ios::binary);
  if (!ifstream.is_open()) {
    std::fprintf(stderr, "could not open %s\n", fname);
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

// not very clean, but works as long as sizeof(int) == sizeof(float)
uint32_t* ivecs_read(const char* fname, size_t &d_out, size_t &n_out) {
  return (uint32_t*) fvecs_read(fname, d_out, n_out);
}