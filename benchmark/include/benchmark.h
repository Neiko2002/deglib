#pragma once

#include <filesystem>

#include "deglib.h"
#include "stopwatch.h"

namespace deglib::benchmark
{

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

static float test_approx(const deglib::search::SearchGraph& graph, const std::vector<uint32_t>& entry_node_indizies,
                         const deglib::FeatureRepository& query_repository, const std::vector<tsl::robin_set<uint32_t>>& ground_truth, 
                         const float eps, const int k)
{
    size_t total = 0;
    size_t correct = 0;
    for (int i = 0; i < query_repository.size(); i++)
    {
        auto query = reinterpret_cast<const std::byte*>(query_repository.getFeature(i));
        auto result_queue = graph.yahooSearch(entry_node_indizies, query, eps, k);

        const auto gt = ground_truth[i];
        total += gt.size();
        while (result_queue.empty() == false)
        {
            const auto internal_index = result_queue.top().getInternalIndex();
            const auto external_id = graph.getExternalLabel(internal_index);
            if (gt.find(external_id) != gt.end()) correct++;
            result_queue.pop();
        }
    }

    return 1.0f * correct / total;
}

static void test_vs_recall(const deglib::search::SearchGraph& graph, const std::vector<uint32_t>& entry_node_indizies, const deglib::FeatureRepository& query_repository,
                           const std::vector<tsl::robin_set<uint32_t>>& ground_truth, const uint32_t k)
{
    // try different eps values for the search radius
    std::vector<float> eps_parameter = {0.1, 0.12, 0.14, 0.16};
    for (float eps : eps_parameter)
    {
        StopW stopw = StopW();
        float recall = deglib::benchmark::test_approx(graph, entry_node_indizies, query_repository, ground_truth, eps, k);
        uint64_t time_us_per_query = stopw.getElapsedTimeMicro() / query_repository.size();

        fmt::print("eps {} \t recall {} \t time_us_per_query {}us\n", eps, recall, time_us_per_query);
        if (recall > 1.0)
        {
            fmt::print("recall {} \t time_us_per_query {}us\n", recall, time_us_per_query);
            break;
        }
    }
}

static void test_graph(const deglib::search::SearchGraph& graph, const deglib::FeatureRepository& query_repository, const uint32_t* ground_truth)
{
    // reproduceable entry point for the graph search
    const uint32_t entry_node_id = 0;
    const auto entry_node_indizies = std::vector<uint32_t> { graph.getInternalIndex(entry_node_id) };
    fmt::print("internal id {} \n", graph.getInternalIndex(entry_node_id));

    // test ground truth
    uint32_t k = 100;  // k at test time
    fmt::print("Parsing gt:\n");
    auto answer = get_ground_truth(ground_truth, query_repository.size(), k);
    fmt::print("Loaded gt:\n");
    for (int i = 0; i < 1; i++) 
        deglib::benchmark::test_vs_recall(graph, entry_node_indizies, query_repository, answer, k);
    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
}

static auto load_sift1m_readonly_graph(std::filesystem::path data_path, std::filesystem::path graph_file) 
{
    const auto path_repository = (data_path / "SIFT1M/sift_base.fvecs").string();
    auto repository = deglib::load_repository(path_repository.c_str());
    fmt::print("{} Base Features with {} dimensions \n", repository.size(), repository.dims());

    StopW stopw = StopW();
    const auto path_graph = (data_path / graph_file).string();
    auto graph = deglib::graph::load_readonly_graph(path_graph.c_str(), repository);
    uint64_t time_in_ms = stopw.getElapsedTimeMicro() / 1000;
    fmt::print("graph node count {} took {}ms\n", graph.size(), time_in_ms);

    return graph;
}

static auto load_sift1m_sizebounded_graph(std::filesystem::path data_path, std::filesystem::path graph_file) 
{
    const auto path_repository = (data_path / "SIFT1M/sift_base.fvecs").string();
    auto repository = deglib::load_repository(path_repository.c_str());
    fmt::print("{} Base Features with {} dimensions \n", repository.size(), repository.dims());

    StopW stopw = StopW();
    const auto path_graph = (data_path / graph_file).string();
    auto graph = deglib::graph::load_sizebounded_graph(path_graph.c_str(), repository);
    uint64_t time_in_ms = stopw.getElapsedTimeMicro() / 1000;
    fmt::print("graph node count {} took {}ms\n", graph.size(), time_in_ms);

    return graph;
}

}  // namespace deglib::benchmark