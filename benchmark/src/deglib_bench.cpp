#include "deglib.h"
#include "stopwatch.h"

#include <fmt/core.h>
#include <tsl/robin_set.h>

// not very clean, but works as long as sizeof(int) == sizeof(float)
uint32_t* ivecs_read(const char* fname, size_t& d_out, size_t& n_out)
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

static float test_approx(const deglib::Graph& graph, deglib::FeatureRepository& repository,
                         const std::vector<uint32_t>& entry_node_ids,
                         const deglib::FeatureRepository& query_repository,
                         const std::vector<tsl::robin_set<uint32_t>>& ground_truth, const deglib::L2Space& l2space,
                         const float eps, const int k)
{
    size_t total = 0;
    size_t correct = 0;
    for (int i = 0; i < query_repository.size(); i++)
    {
        const auto gt = ground_truth[i];
        const auto query = query_repository.getFeature(i);
        auto result_queue = deglib::yahooSearch(graph, repository, entry_node_ids, query, l2space, eps, k);

        total += gt.size();
        while (result_queue.empty() == false)
        {
            if (gt.find(result_queue.top().getId()) != gt.end()) correct++;
            result_queue.pop();
        }
    }

    return 1.0f * correct / total;
}

static void test_vs_recall(const deglib::Graph& graph, deglib::FeatureRepository& repository,
                           const deglib::FeatureRepository& query_repository,
                           const std::vector<tsl::robin_set<uint32_t>>& ground_truth, const deglib::L2Space& l2space,
                           const uint32_t k)
{
    // reproduceable entry point for the graph search
    auto entry_node_ids = std::vector<uint32_t> {0};

    // try different eps values for the search radius
    std::vector<float> eps_parameter = {0.1, 0.12, 0.14};
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

static void test_graph(deglib::Graph& graph, deglib::FeatureRepository& repository, deglib::FeatureRepository& query_repository, uint32_t *ground_truth)
{
    const auto l2space = deglib::L2Space(repository.dims());

    // reproduceable entry point for the graph search
    auto entry_node_ids = std::vector<uint32_t> {0};
    /*auto entry_node_ids = std::vector<uint32_t>();
    entry_node_ids.reserve(1);
    auto it = repository.begin();
    if (it != repository.end()) 
        entry_node_ids.push_back(it->first);*/

    // reproduceable query for the graph search
    {
        auto query = query_repository.getFeature(0);
        int k = 10;
        float eps = 0.1;
        auto result_queue = deglib::yahooSearch(graph, repository, entry_node_ids, query, l2space, eps, k);

        while (result_queue.size() > 0)
        {
            auto entry = result_queue.top();
            result_queue.pop();
            fmt::print("entry id {} and distance {} \n", entry.getId(), entry.getDistance());
        }
    }

    // test ground truth
    uint32_t k = 100;  // k at test time
    fmt::print("Parsing gt:\n");
    auto answer = get_ground_truth(ground_truth, query_repository.size(), k);
    fmt::print("Loaded gt:\n");
    for (int i = 0; i < 1; i++) 
        test_vs_recall(graph, repository, query_repository, answer, l2space, k);
    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
}


static float test_approx_static(const deglib::StaticGraph& graph, deglib::FeatureRepository& repository,
                         const std::vector<uint32_t>& entry_node_ids,
                         const deglib::FeatureRepository& query_repository,
                         const std::vector<tsl::robin_set<uint32_t>>& ground_truth, const deglib::L2Space& l2space,
                         const float eps, const int k)
{
    size_t total = 0;
    size_t correct = 0;
    for (int i = 0; i < query_repository.size(); i++)
    {
        const auto gt = ground_truth[i];
        const auto query = query_repository.getFeature(i);
        auto result_queue = deglib::yahooSearchStatic(graph, repository, entry_node_ids, query, l2space, eps, k);

        total += gt.size();
        while (result_queue.empty() == false)
        {
            if (gt.find(result_queue.top().getId()) != gt.end()) correct++;
            result_queue.pop();
        }
    }

    return 1.0f * correct / total;
}

static void test_vs_recall_static(const deglib::StaticGraph& graph, deglib::FeatureRepository& repository,
                           const deglib::FeatureRepository& query_repository,
                           const std::vector<tsl::robin_set<uint32_t>>& ground_truth, const deglib::L2Space& l2space,
                           const uint32_t k)
{
    // reproduceable entry point for the graph search
    auto entry_node_ids = std::vector<uint32_t> {0};

    // try different eps values for the search radius
    std::vector<float> eps_parameter = {0.1, 0.12, 0.14};
    for (float eps : eps_parameter)
    {
        StopW stopw = StopW();
        float recall = test_approx_static(graph, repository, entry_node_ids, query_repository, ground_truth, l2space, eps, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / query_repository.size();

        fmt::print("eps {} \t recall {} \t time_us_per_query {}us\n", eps, recall, time_us_per_query);
        if (recall > 1.0)
        {
            fmt::print("recall {} \t time_us_per_query {}us\n", recall, time_us_per_query);
            break;
        }
    }
}

static void test_static_graph(deglib::StaticGraph& graph, deglib::FeatureRepository& repository, deglib::FeatureRepository& query_repository, uint32_t *ground_truth)
{
    const auto l2space = deglib::L2Space(repository.dims());

    // test ground truth
    uint32_t k = 100;  // k at test time
    fmt::print("Parsing gt:\n");
    auto answer = get_ground_truth(ground_truth, query_repository.size(), k);
    fmt::print("Loaded gt:\n");
    for (int i = 0; i < 1; i++) 
        test_vs_recall_static(graph, repository, query_repository, answer, l2space, k);
    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
}


int main()
{
    fmt::print("Testing  ...\n");

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif

    auto data_path = std::filesystem::path(DATA_PATH);
    fmt::print("Data dir  {} \n", data_path.string().c_str());

    const auto path_repository = (data_path / "SIFT1M/sift_base.fvecs").string();
    auto repository = deglib::load_static_repository(path_repository.c_str());
    fmt::print("{} Base Features with {} dimensions \n", repository.size(), repository.dims());

    
    const auto path_graph =
        (data_path / "k24nns_128D_L2_Path10_Rnd3+3Improve_AddK20Eps0.2_ImproveK20Eps0.025_WorstEdge0_cpp.graph")
            .string();
    auto static_graph = deglib::load_static_graph(path_graph.c_str(), repository);
    fmt::print("static graph node count {} \n", static_graph.size());

    /*
    StopW stopw = StopW();
    auto graph = deglib::load_graph(path_graph.c_str(), repository);
    float time_in_ms = stopw.getElapsedTimeMicro() / 1000;
    fmt::print("graph node count {} took {}ms\n", graph.size(), time_in_ms);
    */

    const auto path_query_repository = (data_path / "SIFT1M/sift_query.fvecs").string();
    auto query_repository = deglib::load_static_repository(path_query_repository.c_str());
    fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());

    const auto path_query_groundtruth = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    size_t dims;
    size_t count;
    auto ground_truth = ivecs_read(path_query_groundtruth.c_str(), dims, count);
    fmt::print("{} ground truth {} dimensions \n", count, dims);

    //test_graph(graph, repository, query_repository, ground_truth);
    test_static_graph(static_graph, repository, query_repository, ground_truth);

    fmt::print("Test OK\n");
    return 0;
}