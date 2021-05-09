#include <fmt/core.h>
#include <tsl/robin_set.h>

#include "deglib.h"
#include "stopwatch.h"

// not very clean, but works as long as sizeof(int) == sizeof(float)
uint32_t* ivecs_read(const char* fname, size_t &d_out, size_t &n_out) {
  return (uint32_t*)deglib::fvecs_read(fname, d_out, n_out);
}

static std::vector<tsl::robin_set<uint32_t>> get_ground_truth(const uint32_t *ground_truth, const size_t ground_truth_size, const size_t k) {

    auto answers = std::vector<tsl::robin_set<uint32_t>>();
    answers.reserve(ground_truth_size);
    for (int i = 0; i < ground_truth_size; i++) {

        auto gt = tsl::robin_set<uint32_t>();
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) 
          gt.insert(ground_truth[k * i + j]);
        
        answers.push_back(gt);
    }

    return answers;
}

static float test_approx(const deglib::Graph &graph, const deglib::DynamicFeatureRepository &repository, const std::vector<uint32_t> &entry_node_ids, const deglib::StaticFeatureRepository &query_repository, const std::vector<tsl::robin_set<uint32_t>> &ground_truth, const deglib::L2Space &l2space, const float eps, const int k) {

  size_t correct = 0;
  size_t total = 0;
  for (int i = 0; i < query_repository.size(); i++) {
    auto gt = ground_truth[i];
    auto query = query_repository.getFeature(i);
    auto result_queue = deglib::yahooSearch(graph, repository, entry_node_ids, query, l2space, eps, k);

    total += gt.size();
    while (result_queue.empty() == false) {
      if (gt.find(result_queue.top().getId()) != gt.end()) 
        correct++;
      result_queue.pop();
    }
  }

  return 1.0f * correct / total;
}

static void test_vs_recall(const deglib::Graph &graph, const deglib::DynamicFeatureRepository &repository, deglib::StaticFeatureRepository &query_repository, const std::vector<tsl::robin_set<uint32_t>> &ground_truth, const deglib::L2Space &l2space, const int k) {

  // reproduceable entry point for the graph search
  auto entry_node_ids = std::vector<uint32_t>();
  entry_node_ids.reserve(1);
  const auto it = repository.cbegin();
  if(it != repository.cend())
    entry_node_ids.push_back(it.key());

  // 
  std::vector<float> eps_parameter = { 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3 };
  for (float eps : eps_parameter)
  {
    StopW stopw = StopW();
    float recall = test_approx(graph, repository, entry_node_ids, query_repository, ground_truth, l2space, eps, k);
    float time_us_per_query = stopw.getElapsedTimeMicro() / query_repository.size();

    fmt::print("eps {} \t recall {} \t time_us_per_query {}us\n", eps, recall, time_us_per_query);
    if (recall > 1.0)
    {
      fmt::print("recall {} \t time_us_per_query {}us\n", recall, time_us_per_query);
      break;
    }
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
  {
    auto query = query_repository.getFeature(0);
    int k = 10;
    float eps = 0.1;
    auto result_queue = deglib::yahooSearch(graph, repository, entry_node_ids, query, l2space, eps, k);

    while(result_queue.size() > 0) {
      auto entry = result_queue.top();
      result_queue.pop();
      fmt::print("entry id {} and distance {} \n", entry.getId(), entry.getDistance());
    }
  }

  // test ground truth
  size_t k = 100; // k at test time
  fmt::print("Parsing gt:\n");
  auto answer = get_ground_truth(ground_truth, query_repository.size(), k);
  fmt::print("Loaded gt:\n");
  for (int i = 0; i < 1; i++)
    test_vs_recall(graph, repository, query_repository, answer, l2space, k);
  fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);

  fmt::print("Test OK\n");
  return 0;
}