#pragma once

#include <filesystem>

#include "deglib.h"
#include "stopwatch.h"

namespace deglib::benchmark
{

static std::vector<tsl::robin_set<uint32_t>> get_ground_truth(const uint32_t* ground_truth, const size_t ground_truth_size, const uint32_t ground_truth_dims, const size_t k)
{
    auto answers = std::vector<tsl::robin_set<uint32_t>>(ground_truth_size);
    for (uint32_t i = 0; i < ground_truth_size; i++)
    {
        auto& gt = answers[i];
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) 
            gt.insert(ground_truth[ground_truth_dims * i + j]);
    }

    return answers;
}

static float test_approx_anns(const deglib::search::SearchGraph& graph, const std::vector<uint32_t>& entry_node_indizies,
                         const deglib::FeatureRepository& query_repository, const std::vector<tsl::robin_set<uint32_t>>& ground_truth, 
                         const float eps, const uint32_t k)
{
    size_t total = 0;
    size_t correct = 0;
    for (uint32_t i = 0; i < uint32_t(query_repository.size()); i++)
    {
        auto query = reinterpret_cast<const std::byte*>(query_repository.getFeature(i));
        auto result_queue = graph.yahooSearch(entry_node_indizies, query, eps, k);

        total += result_queue.size();
        const auto gt = ground_truth[i];
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

static float test_approx_explore(const deglib::search::SearchGraph& graph, const std::vector<std::vector<uint32_t>>& entry_node_indizies, 
                                 const deglib::FeatureRepository& query_repository, const std::vector<tsl::robin_set<uint32_t>>& ground_truth, 
                                 const float eps, const uint32_t k)
{    
    size_t total = 0;
    size_t correct = 0;
    for (uint32_t i = 0; i < query_repository.size(); i++) {
        const auto& entry_nodes = entry_node_indizies[i];

        auto query = reinterpret_cast<const std::byte*>(query_repository.getFeature(i));
        auto result_queue = graph.yahooSearch(entry_nodes, query, eps, k);

        total += k;
        const auto& gt = ground_truth[i];
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

static void test_graph_anns(const deglib::search::SearchGraph& graph, const deglib::FeatureRepository& query_repository, const uint32_t* ground_truth, const uint32_t ground_truth_dims, const uint32_t repeat, const uint32_t k)
{
    // reproduceable entry point for the graph search
    const uint32_t entry_node_id = 0;
    const auto entry_node_indizies = std::vector<uint32_t> { graph.getInternalIndex(entry_node_id) };
    fmt::print("internal id {} \n", graph.getInternalIndex(entry_node_id));

    // test ground truth
    fmt::print("Parsing gt:\n");
    auto answer = deglib::benchmark::get_ground_truth(ground_truth, query_repository.size(), ground_truth_dims, k);
    fmt::print("Loaded gt:\n");

    // try different eps values for the search radius
    std::vector<float> eps_parameter = { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f };
    for (float eps : eps_parameter)
    {
        StopW stopw = StopW();
        float recall = 0;

        for (size_t i = 0; i < repeat; i++) 
            recall = deglib::benchmark::test_approx_anns(graph, entry_node_indizies, query_repository, answer, eps, k);
        uint64_t time_us_per_query = (stopw.getElapsedTimeMicro() / query_repository.size()) / repeat;

        fmt::print("eps {} \t recall {} \t time_us_per_query {}us\n", eps, recall, time_us_per_query);
        if (recall > 1.0)
            break;
    }

    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
}

static void test_graph_explore(const deglib::search::SearchGraph& graph, const deglib::FeatureRepository& query_repository, const uint32_t* ground_truth, const uint32_t ground_truth_dims, const uint32_t* entry_nodes, const uint32_t entry_node_size, const uint32_t repeat, const uint32_t k)
{  fmt::print("Explore for {} neighbors\n", k);

    if (ground_truth_dims < k)
    {
        fmt::print(stderr, "ground thruth data does not have enough dimensions, expected {} got {} \n", k, ground_truth_dims);
        perror("");
        abort();
    }
    
    // test ground truth
    fmt::print("Parsing gt:\n");
    auto answer = deglib::benchmark::get_ground_truth(ground_truth, query_repository.size(), ground_truth_dims, k);
    fmt::print("Loaded gt:\n");

    // reproduceable entry point for the graph search
    auto entry_node_indizies = std::vector<std::vector<uint32_t>>();
    for (size_t i = 0; i < query_repository.size(); i++) {
        auto entry_node = std::vector<uint32_t>(entry_nodes + i * entry_node_size, entry_nodes + (i+1) * entry_node_size);
        entry_node_indizies.emplace_back(entry_node);
    }

    // try different eps values for the search radius
    std::vector<float> eps_parameter = { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f };
    for (float eps : eps_parameter)
    {
        StopW stopw = StopW();
        float recall = 0;

        for (size_t i = 0; i < repeat; i++) 
            recall = deglib::benchmark::test_approx_explore(graph, entry_node_indizies, query_repository, answer, eps, k);
        uint64_t time_us_per_query = (stopw.getElapsedTimeMicro() / query_repository.size()) / repeat;

        fmt::print("eps {} \t recall {} \t time_us_per_query {}us\n", eps, recall, time_us_per_query);
        if (recall > 1.0)
            break;
    }

    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
}

}  // namespace deglib::benchmark