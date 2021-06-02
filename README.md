# DEG - Dynamic Exploration Graphs for efficient approximate nearest neighbor search
Header-only C++ DEG implementation with SIFT1M experiment.

### Introduction
This repository contains the necessary code to build a Dynamic Exploration Graph (DEG) and use it for efficient approximate nearest neighbor search (ANNS) tasks. Compared to other algorithms in the litature, DEG changes over time by swapping edges to yield better search results in the future. In case of a static dataset like the [SIFT1M](http://corpus-texmex.irisa.fr/) the graph gets slowly extended and improved step by step until all the data is stored inside the graph. Afterwards it can be tested with the query and ground truth information of the dataset. The results of those tests show DEG is on-par with the other state-of-the-art algorithms. But its real strength comes to shine in dynamic environments. In use-cases where the data changes over time (like products of a shopping website) DEG can always provide the best search results without recalculating the entire graph.

### Highlights
1) Lightweight, header-only, right now only minor dependencies (all header-only) and compliend to C++ 20.
2) Has full support for incremental graph construction and improvement. 
3) Can work with custom user defined distance metricies (C++).
4) Has a comparison branch with the [HNSW library](https://github.com/nmslib/hnswlib)

### SIFT1M benchmark reproduction
The code for the benchmark is in the [deglib_build_bench.cpp](https://github.com/Neiko2002/deglib/blob/hnswlib_deglib/benchmark/src/deglib_build_bench.cpp "deglib_build_bench.cpp") and [hnswlib_bench.cpp](https://github.com/Neiko2002/deglib/blob/hnswlib_deglib/benchmark/src/hnswlib_bench.cpp "hnswlib_bench.cpp") file. The dataset must be downloaded before hand. Both benchmark files contain a DATA_PATH variable which can be changed by renaming [cmake-variants.sample.yaml](https://github.com/Neiko2002/deglib/blob/hnswlib_deglib/cmake-variants.sample.yaml "cmake-variants.sample.yaml") file to **cmake-variants.yaml** and add the path to the dataset in the last line.

The DEG builder benchmark has a lot of parameters. Generally we recommend changing only the k and eps parameter and keep it the same for all variants (extend, improve, improve_extended). To quickly reproduce the results in the line chart below the following settings are needed:
edges_per_node = 24;
extend_k = 20; 
extend_eps = 0.2;
improve_k = 20;
improve_eps = 0.02;
improve_extended_k = 12;
improve_extended_eps = 0.02;
max_path_length = 10;
swap_tries = 3;
additional_swap_tries = 3;

For the HNSW graph it will be the following settings:
efConstruction = 500;
M = 24;

![line chart showing the ANNS timing for the SIFT1M benchmark](https://raw.githubusercontent.com/Neiko2002/deglib/hnswlib_deglib/results/sift1m.svg)

### Roadmap
There is no specific order. All the functionality listed here will be added when the associated paper comes out.
- Support for element deletions will be added in the future
- Benchmark for dynamic datasets
- Python bindings


### References

Coming soon
