#include <random>
#include <chrono>
#include <omp.h>

#include <fmt/core.h>

#include "benchmark.h"
#include "deglib.h"


/**
 * Convert the queue into a vector with ascending distance order
 **/
static auto topListAscending(deglib::search::ResultSet& queue) {
  const auto size = (int32_t) queue.size();
  auto topList = std::vector<deglib::search::ObjectDistance>(size);
  for (int32_t i = size - 1; i >= 0; i--) {
    topList[i] = std::move(const_cast<deglib::search::ObjectDistance&>(queue.top()));
    queue.pop();
  }
  return topList;
}

/**
 * Extend the graph with a new vertex. Find good existing vertex to which this new vertex gets connected.
 */
void extendGraph(deglib::graph::MutableGraph& graph, const uint32_t external_label, std::vector<std::byte> new_vertex_feature, std::mt19937& rnd_, bool schemaC, float extend_eps_, uint32_t extend_k_) {
 
  // graph should not contain a vertex with the same label
  if(graph.hasNode(external_label)) {
    fmt::print(stderr, "graph contains vertex {} already. can not add it again \n", external_label);
    perror("");
    abort();
  }

  // fully connect all vertices
  const auto edges_per_vertex = uint32_t(graph.getEdgesPerNode());
  if(graph.size() < edges_per_vertex+1) {

    // add an empty vertex to the graph (no neighbor information yet)
    const auto internal_index = graph.addNode(external_label, new_vertex_feature.data());

    // connect the new vertex to all other vertices in the graph
    const auto dist_func = graph.getFeatureSpace().get_dist_func();
    const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
    for (size_t i = 0; i < graph.size(); i++) {
      if(i != internal_index) {
        const auto dist = dist_func(new_vertex_feature.data(), graph.getFeatureVector(i), dist_func_param);
        graph.changeEdge(i, i, internal_index, dist);
        graph.changeEdge(internal_index, internal_index, i, dist);
      }
    }
    return;
  }

  // find good neighbors for the new vertex
  auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
  const std::vector<uint32_t> entry_vertex_indices = { distrib(rnd_) };
  auto top_list = graph.search(entry_vertex_indices, new_vertex_feature.data(), extend_eps_, std::max(uint32_t(extend_k_), edges_per_vertex));
  const auto results = topListAscending(top_list);

  // their should always be enough neighbors (search results), otherwise the graph would be broken
  if(results.size() < edges_per_vertex) {
    fmt::print(stderr, "the graph search for the new vertex {} did only provided {} results \n", external_label, results.size());
    perror("");
    abort();
  }

  // add an empty vertex to the graph (no neighbor information yet)
  const auto internal_index = graph.addNode(external_label, new_vertex_feature.data());

  // for computing distances to neighbors not in the result queue
  const auto dist_func = graph.getFeatureSpace().get_dist_func();
  const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
 
  // adding neighbors happens in two phases, the first tries to retain RNG, the second adds them without checking
  int check_rng_phase = 1; // 1 = activated, 2 = deactived

  // remove an edge of the good neighbors and connect them with this new vertex
  auto new_neighbors = std::vector<std::pair<uint32_t, float>>();
  while(new_neighbors.size() < edges_per_vertex) {
    for (size_t i = 0; i < results.size() && new_neighbors.size() < edges_per_vertex; i++) {
      const auto candidate_index = results[i].getInternalIndex();
      const auto candidate_weight = results[i].getDistance();
      const auto& result = results[i];

      // check if the vertex is already in the edge list of the new vertex (added during a previous loop-run)
      // since all edges are undirected and the edge information of the new vertex does not yet exist, we search the other way around.
      if(graph.hasEdge(result.getInternalIndex(), internal_index)) 
        continue;

      // does this vertex has a neighbor which is connected to the new vertex and has a lower distance?
      if(check_rng_phase <= 1 && deglib::analysis::checkRNG(graph, edges_per_vertex, candidate_index, internal_index, candidate_weight) == false) 
        continue;

      // This version is good for high LID datasets or small graphs with low distance count limit during ANNS
      uint32_t new_neighbor_index = 0;
      float new_neighbor_distance = -1;      
      if(schemaC) {

        // find the worst edge of the new neighbor
        float new_neighbor_weight = -1;                                      // version C in the paper
        const auto neighbor_weights = graph.getNeighborWeights(result.getInternalIndex());
        const auto neighbor_indices = graph.getNeighborIndices(result.getInternalIndex());
        for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
          const auto neighbor_index = neighbor_indices[edge_idx];

          // the suggested neighbor might already be in the edge list of the new vertex
          if(graph.hasEdge(neighbor_index, internal_index))
            continue;

          // find a non-RNG edge between the candidate_index and neighbor_index, which would be RNG conform between internal_index and neighbor_index
          const auto neighbor_weight = neighbor_weights[edge_idx];

          // the suggested neighbor might already be in the edge list of the new vertex
          // the weight of the neighbor might not be worst than the current worst one         
          if(neighbor_weight > new_neighbor_weight) {       // version C in the paper
            new_neighbor_weight = neighbor_weight;
            new_neighbor_index = neighbor_index;
          }       
        }
        // new_neighbor_weight == -1 should not be possible, otherwise the new vertex is connected to every vertex in the neighbor-list of the result-vertex and still has space for more
        if(new_neighbor_weight == -1) 
          continue;
        
        new_neighbor_distance = dist_func(new_vertex_feature.data(), graph.getFeatureVector(new_neighbor_index), dist_func_param); 
      }
      else
      {
        // find the edge which improves the distortion the most: (distance_new_edge1 + distance_new_edge2) - distance_removed_edge       
        float best_distortion = std::numeric_limits<float>::max();
        const auto neighbor_indices = graph.getNeighborIndices(result.getInternalIndex());
        const auto neighbor_weights = graph.getNeighborWeights(result.getInternalIndex());
        for (size_t edge_idx = 0; edge_idx < edges_per_vertex; edge_idx++) {
          const auto neighbor_index = neighbor_indices[edge_idx];
          if(graph.hasEdge(neighbor_index, internal_index) == false) {
            const auto neighbor_distance = dist_func(new_vertex_feature.data(), graph.getFeatureVector(neighbor_index), dist_func_param);

            // take the neighbor with the best distance to the new vertex, which might already be in its edge list
            float distortion = (result.getDistance() + neighbor_distance) - neighbor_weights[edge_idx];   // version D in the paper
            if(distortion < best_distortion) {
              best_distortion = distortion;
              new_neighbor_index = neighbor_index;
              new_neighbor_distance = neighbor_distance;
            }          
          }
        }
      }

      // this should not be possible, otherwise the new vertex is connected to every vertex in the neighbor-list of the result-vertex and still has space for more
      if(new_neighbor_distance == -1) 
        continue;
      
      // place the new vertex in the edge list of the result-vertex
      graph.changeEdge(result.getInternalIndex(), new_neighbor_index, internal_index, result.getDistance());
      new_neighbors.emplace_back(result.getInternalIndex(), result.getDistance());

      // place the new vertex in the edge list of the best edge neighbor
      graph.changeEdge(new_neighbor_index, result.getInternalIndex(), internal_index, new_neighbor_distance);
      new_neighbors.emplace_back(new_neighbor_index, new_neighbor_distance);
    }
    check_rng_phase++;
  }

  if(new_neighbors.size() < edges_per_vertex) {
    fmt::print(stderr, "could find only {} good neighbors for the new vertex {} need {}\n", new_neighbors.size(), internal_index, edges_per_vertex);
    perror("");
    abort();
  }

  // sort the neighbors by their neighbor indices and store them in the new vertex
  {
    std::sort(new_neighbors.begin(), new_neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
    auto neighbor_indices = std::vector<uint32_t>(new_neighbors.size());
    auto neighbor_weights = std::vector<float>(new_neighbors.size());
    for (size_t i = 0; i < new_neighbors.size(); i++) {
      const auto& neighbor = new_neighbors[i];
      neighbor_indices[i] = neighbor.first;
      neighbor_weights[i] = neighbor.second;
    }
    graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());  
  }
}


/**
 * Load the graph from the drive and test it against the SIFT query data.
 */
void test_graph(const std::string path_query_repository, const std::string path_query_groundtruth, const std::string graph_file, const uint32_t repeat, const uint32_t k) {


    // load an existing graph
    fmt::print("Load graph {} \n", graph_file);
    const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading the graph\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);
    

    const auto query_repository = deglib::load_static_repository(path_query_repository.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());

    size_t dims_out;
    size_t count_out;
    const auto ground_truth_f = deglib::fvecs_read(path_query_groundtruth.c_str(), dims_out, count_out);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("{} ground truth {} dimensions \n", count_out, dims_out);

    deglib::benchmark::test_graph_anns(graph, query_repository, ground_truth, (uint32_t)dims_out, repeat, k);
}


int main() {

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb \n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    omp_set_num_threads(1);
    std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;

    const auto data_path        = std::filesystem::path(DATA_PATH);
    uint32_t repeat_test = 1;
    uint32_t test_k = 100;


    // ---------------------------- Deep10M ------------------------------
    const auto repository_file  = (data_path / "deep10m" / "deep10m_base.fvecs").string();
    const auto query_file       = (data_path / "deep10m" / "deep10m_query.fvecs").string();
    const auto groundtruth_file = (data_path / "deep10m" / "deep10m_groundtruth.ivecs").string();
    const auto graph_file       = (data_path / "deg" / "96D_L2_K60_AddK60Eps0.1High_schemaD.deg").string();

    // build
    if(std::filesystem::exists(graph_file.c_str()) == false) {
        auto rnd = std::mt19937(7);  // default 7
        const int weight_scale = 100; // SIFT+enron+crawl=1 UQ-V=100000 Glove=1000 CLIP=1000
        const uint8_t edges_per_vertex = 60;
        const deglib::Metric metric = deglib::Metric::L2;
        const uint8_t extend_k = 60; // should be greater or equals to edges_per_vertex
        const float extend_eps = 0.0f;
        const bool schema_c = false; // true=SchemaC false=SchemaD

        // load data
        fmt::print("Load Data \n");
        auto repository = deglib::load_static_repository(repository_file.c_str());   
        fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after loading data\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        // create a new graph
        fmt::print("Setup empty graph with {} vertices in {}D feature space\n", repository.size(), repository.dims());
        const auto dims = repository.dims();
        const uint32_t max_vertex_count = uint32_t(repository.size());
        const auto feature_space = deglib::FloatSpace(dims, metric);
        auto graph = deglib::graph::SizeBoundedGraph(max_vertex_count, edges_per_vertex, feature_space);
        fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb after setup empty graph\n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        // create a graph builder to add vertices to the new graph and improve its edges
        fmt::print("Start graph builder \n");   
    
        // provide all features to the graph builder at once. In an online system this will be called 
        const auto base_size = uint32_t(repository.size());
        const auto log_after = base_size / 10;
        auto start = std::chrono::steady_clock::now();
        uint64_t duration_ms = 0;
        for (uint32_t i = 0; i < base_size; i++) { 

            auto label = i;
            auto feature = reinterpret_cast<const std::byte*>(repository.getFeature(label));
            auto feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};

            extendGraph(graph, label, feature_vector, rnd, schema_c, extend_eps, extend_k);

            const auto size = graph.size();
            if(size % log_after == 0 || size == base_size) {

                duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
                auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, weight_scale); 
                auto weight_histogram_sorted = deglib::analysis::calc_edge_weight_histogram(graph, true, weight_scale);
                auto weight_histogram = deglib::analysis::calc_edge_weight_histogram(graph, false, weight_scale);
                auto valid_weights = deglib::analysis::check_graph_validation(graph, uint32_t(size), true);
                auto connected = deglib::analysis::check_graph_connectivity(graph);
                auto duration = duration_ms / 1000;
                auto currRSS = getCurrentRSS() / 1000000;
                auto peakRSS = getPeakRSS() / 1000000;
                fmt::print("{:7} vertices, {:5}s, AEW: {:4.2f} -> Sorted:{:.1f}, InOrder:{:.1f}, {} connected & {}, RSS {} & peakRSS {}\n", 
                            size, duration, avg_edge_weight, fmt::join(weight_histogram_sorted, " "), fmt::join(weight_histogram, " "), connected ? "" : "not", valid_weights ? "valid" : "invalid", currRSS, peakRSS);
                start = std::chrono::steady_clock::now();
            }
            else 
            if(size % (log_after/10) == 0) {
                duration_ms += uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
                auto avg_edge_weight = deglib::analysis::calc_avg_edge_weight(graph, weight_scale);
                auto duration = duration_ms / 1000;
                auto currRSS = getCurrentRSS() / 1000000;
                auto peakRSS = getPeakRSS() / 1000000;

                fmt::print("{:7} vertices, {:5}s, AEW: {:4.2f}, RSS {} & peakRSS {}\n", size, duration, avg_edge_weight, currRSS, peakRSS);

                start = std::chrono::steady_clock::now();
            }
        }
        graph.saveGraph(graph_file.c_str());
    }

    // test
    if(std::filesystem::exists(graph_file.c_str()) == true) 
        test_graph(query_file, groundtruth_file, graph_file, repeat_test, test_k);

    fmt::print("Test OK\n");
    return 0;
}