#pragma once

#include <filesystem>
#include <tsl/robin_set.h>

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

static float test_approx_anns(const deglib::search::SearchGraph& graph, const std::vector<uint32_t>& entry_node_indices,
                         const deglib::FeatureRepository& query_repository, const std::vector<tsl::robin_set<uint32_t>>& ground_truth, 
                         const float eps, const uint32_t k)
{
    size_t total = 0;
    size_t correct = 0;
    for (uint32_t i = 0; i < uint32_t(query_repository.size()); i++)
    {
        auto query = reinterpret_cast<const std::byte*>(query_repository.getFeature(i));
        auto result_queue = graph.search(entry_node_indices, query, eps, k);
        // auto result_queue = graph.search(entry_node_indices, query, eps, k, 4); // max distance calcs
        

        if (result_queue.size() != k) {
            fmt::print(stderr, "ANNS with k={} got only {} results for query {}\n", k, result_queue.size(), i);
            abort();
        }

        total += result_queue.size();
        const auto gt = ground_truth[i];
        // auto checked_ids = tsl::robin_set<uint32_t>(); // additional check
        while (result_queue.empty() == false)
        {
            const auto internal_index = result_queue.top().getInternalIndex();
            const auto external_id = graph.getExternalLabel(internal_index);
            if (gt.find(external_id) != gt.end()) correct++;
            result_queue.pop();
            // checked_ids.insert(internal_index);
        }

        // if (checked_ids.size() != k) {
        //     fmt::print(stderr, "ANNS with k={} got only {} unique ids \n", k, checked_ids.size());
        //     abort();
        // }
    }

    return 1.0f * correct / total;
}

static float test_approx_explore(const deglib::search::SearchGraph& graph, const std::vector<std::vector<uint32_t>>& entry_node_indices, 
                                  const std::vector<tsl::robin_set<uint32_t>>& ground_truth, const uint32_t k, const uint32_t max_distance_count)
{    
    size_t total = 0;
    size_t correct = 0;
    for (uint32_t i = 0; i < entry_node_indices.size(); i++) {
        const auto entry_node_index = entry_node_indices[i][0];
        auto result_queue = graph.explore(entry_node_index, k, max_distance_count);

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
    // find vertex closest to the average feature vector
    // uint32_t entry_node_id;
    // {
    //     const auto feature_dims = graph.getFeatureSpace().dim();
    //     const auto graph_size = (uint32_t) graph.size();
    //     auto avg_fv = std::vector<float>(feature_dims);
    //     for (uint32_t i = 0; i < graph_size; i++) {
    //         auto fv = reinterpret_cast<const float*>(graph.getFeatureVector(i));
    //         for (size_t dim = 0; dim < feature_dims; dim++) 
    //             avg_fv[dim] += fv[dim];
    //     }

    //     for (size_t dim = 0; dim < feature_dims; dim++) 
    //         avg_fv[dim] /= graph_size;

    //     const auto seed = std::vector<uint32_t> { graph.getInternalIndex(0) };
    //     auto result_queue = graph.search(seed, reinterpret_cast<const std::byte*>(avg_fv.data()), 0.1f, 30);
    //     entry_node_id = result_queue.top().getInternalIndex();
    // }

    // reproduceable entry point for the graph search
    // const auto entry_node_indices = std::vector<uint32_t> { graph.getInternalIndex(entry_node_id) };
    const auto entry_node_indices = graph.getEntryNodeIndices();
    fmt::print("internal id {} \n", graph.getInternalIndex(entry_node_indices[0]));

    // test ground truth
    fmt::print("Parsing gt:\n");
    auto answer = deglib::benchmark::get_ground_truth(ground_truth, query_repository.size(), ground_truth_dims, k);
    fmt::print("Loaded gt:\n");

    // try different eps values for the search radius
    std::vector<float> eps_parameter = { 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f  };
    // std::vector<float> eps_parameter = { 0.00f, 0.01f, 0.05f, 0.1f, 0.12f, 0.14f, 0.16f, 0.18f, 0.2f, 0.3f, 0.4f, 0.8f };
    //std::vector<float> eps_parameter = { 0.2f, 0.3f, 0.4f, 0.8f }; //
    for (float eps : eps_parameter)
    {
        StopW stopw = StopW();
        float recall = 0;

        for (size_t i = 0; i < repeat; i++) 
            recall = deglib::benchmark::test_approx_anns(graph, entry_node_indices, query_repository, answer, eps, k);
        uint64_t time_us_per_query = (stopw.getElapsedTimeMicro() / query_repository.size()) / repeat;

        fmt::print("eps {} \t recall {} \t time_us_per_query {}us\n", eps, recall, time_us_per_query);
        if (recall > 1.0)
            break;
    }

    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
}

static void test_graph_explore(const deglib::search::SearchGraph& graph, const uint32_t query_count, const uint32_t* ground_truth, const uint32_t ground_truth_dims, const uint32_t* entry_nodes, const uint32_t entry_node_dims, const uint32_t repeat, const uint32_t k)
{
    if (ground_truth_dims < k)
    {
        fmt::print(stderr, "ground thruth data does not have enough dimensions, expected {} got {} \n", k, ground_truth_dims);
        perror("");
        abort();
    }
    
    // reproduceable entry point for the graph search
    auto entry_node_indices = std::vector<std::vector<uint32_t>>();
    for (size_t i = 0; i < query_count; i++) {
        auto entry_node = std::vector<uint32_t>(entry_nodes + i * entry_node_dims, entry_nodes + (i+1) * entry_node_dims);
        entry_node_indices.emplace_back(entry_node);
    }

    // ground truth data
    const auto answer = deglib::benchmark::get_ground_truth(ground_truth, query_count, ground_truth_dims, k);

    // try different k values
    float k_factor = 0.1f;
    for (uint32_t f = 0; f <= 3; f++) {
        k_factor *= 10;
        //for (uint32_t i = (f == 0) ? 0 : 1; i < 10; i++) {
        //    const auto max_distance_count = k + uint32_t(k*k_factor * i);
         for (uint32_t i = 1; i < 14; i++) {
             const auto max_distance_count = i;

            StopW stopw = StopW();
            float recall = 0;
            for (size_t r = 0; r < repeat; r++) 
                recall = deglib::benchmark::test_approx_explore(graph, entry_node_indices, answer, k, max_distance_count);
            uint64_t time_us_per_query = stopw.getElapsedTimeMicro() / (query_count * repeat);

            fmt::print("max_distance_count {}, k {}, recall {}, time_us_per_query {}us\n", max_distance_count, k, recall, time_us_per_query);
            if (recall > 1.0)
                break;
        }
    }

    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Max memory usage: {} Mb\n", getPeakRSS() / 1000000);
}

}  // namespace deglib::benchmark