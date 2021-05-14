#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_set>

#include <fmt/core.h>
#include <tsl/robin_hash.h>
#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include "deglib.h"
#include "stopwatch.h"

// not very clean, but works as long as sizeof(int) == sizeof(float)
static uint32_t* ivecs_read(const char* fname, size_t& d_out, size_t& n_out)
{
    // memory leak
    return (uint32_t*)deglib::fvecs_read(fname, d_out, n_out).release();
}

static std::vector<tsl::robin_set<uint32_t>> get_ground_truth(const uint32_t* ground_truth,
                                                              const size_t ground_truth_size, const size_t k)
{
    auto answers = std::vector<tsl::robin_set<uint32_t>>();
    answers.reserve(ground_truth_size);
    for (int i = 0; i < ground_truth_size; i++)
    {
        auto gt = tsl::robin_set<uint32_t>();
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) gt.insert(ground_truth[k * i + j]);

        answers.push_back(gt);
    }

    return answers;
}

static float test_approx_readonly(const deglib::ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies,
                         const deglib::FeatureRepository& query_repository, const std::vector<tsl::robin_set<uint32_t>>& ground_truth, 
                          const float eps, const int k)
{
    size_t total = 0;
    size_t correct = 0;
    for (int i = 0; i < query_repository.size(); i++)
    {
        auto query = query_repository.getFeature(i);
        auto result_queue = graph.yahooSearch(entry_node_indizies, query, eps, k);

        const auto gt = ground_truth[i];
        total += gt.size();
        while (result_queue.empty() == false)
        {
            const auto internal_index = result_queue.top().getId();
            const auto external_id = graph.getExternalLabel(internal_index);
            if (gt.find(external_id) != gt.end()) correct++;
            result_queue.pop();
        }
    }

    return 1.0f * correct / total;
}

static void test_vs_recall_readonly(const deglib::ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const deglib::FeatureRepository& query_repository,
                           const std::vector<tsl::robin_set<uint32_t>>& ground_truth, const uint32_t k)
{
    // try different eps values for the search radius
    std::vector<float> eps_parameter = {0.1, 0.12, 0.14, 0.16};
    for (float eps : eps_parameter)
    {
        StopW stopw = StopW();
        float recall = test_approx_readonly(graph, entry_node_indizies, query_repository, ground_truth, eps, k);
        float time_us_per_query = static_cast<float>(stopw.getElapsedTimeMicro()) / query_repository.size();

        fmt::print("eps {} \t recall {} \t time_us_per_query {}us\n", eps, recall, time_us_per_query);
        if (recall > 1.0)
        {
            fmt::print("recall {} \t time_us_per_query {}us\n", recall, time_us_per_query);
            break;
        }
    }
}

static void test_readonly_graph(const deglib::ReadOnlyGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const deglib::FeatureRepository& query_repository, const uint32_t* ground_truth)
{
    // test ground truth
    uint32_t k = 100;  // k at test time
    fmt::print("Parsing gt:\n");
    auto answer = get_ground_truth(ground_truth, query_repository.size(), k);
    fmt::print("Loaded gt:\n");
    for (int i = 0; i < 1; i++) 
        test_vs_recall_readonly(graph, entry_node_indizies, query_repository, answer, k);
    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
}

static auto load_graph(std::filesystem::path data_path) 
{
    const auto path_repository = (data_path / "SIFT1M/sift_base.fvecs").string();
    auto repository = deglib::load_repository(path_repository.c_str());
    fmt::print("{} Base Features with {} dimensions \n", repository.size(), repository.dims());

    StopW stopw = StopW();
    const auto path_graph =
          (data_path / "k24nns_128D_L2_Path10_Rnd3+3Improve_AddK20Eps0.2_ImproveK20Eps0.025_WorstEdge0_cpp.graph")
              .string();
    auto graph = deglib::load_readonly_graph(path_graph.c_str(), repository);
    float time_in_ms = static_cast<float>(stopw.getElapsedTimeMicro()) / 1000;
    fmt::print("graph node count {} took {}ms\n", graph.size(), time_in_ms);

    return graph;
}

int main() {
    fmt::print("Testing ...\n");

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif

    auto data_path = std::filesystem::path(DATA_PATH);
    fmt::print("Data dir  {} \n", data_path.string().c_str());
    const auto graph = load_graph(data_path);

    // reproduceable entry point for the graph search
    const uint32_t entry_node_id = 0;
    const auto entry_node_indizies = std::vector<uint32_t> { graph.getInternalIndex(entry_node_id) };
    fmt::print("internal id {} \n", graph.getInternalIndex(entry_node_id));




    const auto path_query_repository = (data_path / "SIFT1M/sift_query.fvecs").string();
    const auto query_repository = deglib::load_static_repository(path_query_repository.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());
    const auto path_query_groundtruth = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    size_t dims;
    size_t count;
    const auto ground_truth = ivecs_read(path_query_groundtruth.c_str(), dims, count);
    fmt::print("{} ground truth {} dimensions \n", count, dims);
    test_readonly_graph(graph, entry_node_indizies, query_repository, ground_truth);




    fmt::print("Test OK\n");
    return 0;
}