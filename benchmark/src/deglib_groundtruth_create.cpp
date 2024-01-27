#include <fmt/core.h>

#include <omp.h>
#include <random>
#include <math.h>
#include <limits>
#include <algorithm>
#include <tsl/robin_set.h>
#include <tsl/robin_map.h>

#include "benchmark.h"
#include "deglib.h"

/**
 * Convert the queue into a vector with ascending distance order
 **/
static auto topListAscending(deglib::search::ResultSet& queue) {
    const auto size = (int32_t) queue.size();
    auto topList = std::vector<deglib::search::ObjectDistance>(size);
    for(int32_t i = size - 1; i >= 0; i--) {
        topList[i] = std::move(const_cast<deglib::search::ObjectDistance&>(queue.top()));
        queue.pop();
    }
    return topList;
}

/**
 * Write ivecs file
 **/
void ivecs_write(const char *fname, uint32_t d, size_t n, const uint32_t* v) {
    auto out = std::ofstream(fname, std::ios::out | std::ios::binary);

    // check open file for write
    if (!out.is_open()) {
        fmt::print(stderr, "Error in open file {}\n", fname);
        perror("");
        abort();
    }
    for (uint32_t i = 0; i < n; i++) {
        const auto ptr = v + i * d;
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));    
        out.write(reinterpret_cast<const char*>(ptr), sizeof(uint32_t) * d);    
    }

    out.close();
}

/**
 * Write fvecs file
 **/
void fvecs_write(const char *fname, uint32_t d, size_t n, const float* v) {
    auto out = std::ofstream(fname, std::ios::out | std::ios::binary);

    // check open file for write
    if (!out.is_open()) {
        fmt::print(stderr, "Error in open file {}\n", fname);
        perror("");
        abort();
    }
    for (uint32_t i = 0; i < n; i++) {
        const auto ptr = v + i * d;
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));    
        out.write(reinterpret_cast<const char*>(ptr), sizeof(float) * d);    
    }

    out.close();
}

/**
 * Check if the gt_file contains the right information
 */
static void check_gt(const deglib::StaticFeatureRepository& base_repo, const deglib::StaticFeatureRepository& query_repo, const char *gt_file) {
    const auto base_size = (uint32_t)base_repo.size();
    const auto query_size = (uint32_t)query_repo.size();
    const auto dims = base_repo.dims();

    const deglib::Metric metric = deglib::Metric::L2;
    const auto feature_space = deglib::FloatSpace(dims, metric);
    const auto dist_func = feature_space.get_dist_func();
    const auto dist_func_param = feature_space.get_dist_func_param();

    std::cout << "Load Query Data" << std::endl;
    size_t top_k;
    size_t gt_size;
    const auto ground_truth_f = deglib::fvecs_read(gt_file, top_k, gt_size);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    if (gt_size != query_size) {
        fmt::print(stderr, "Ground truth data contains more data than the query repo. Got {} expected {}\n", gt_size, query_size);
        perror("");
        abort();
    }

    for (int q = 0; q < (int)query_size; q++) {
        const auto query = query_repo.getFeature(q);

        auto worst_distance = std::numeric_limits<float>::max();
        auto results = deglib::search::ResultSet(); 
        for (uint32_t b = 0; b < base_size; b++) {
            const auto distance = dist_func(query, base_repo.getFeature(b), dist_func_param);
            if(distance < worst_distance) {
                results.emplace(b, distance);
                if (results.size() > top_k) {
                    results.pop();
                    worst_distance = results.top().getDistance();
                }
            }
        }

        if (results.size() != top_k) {
            fmt::print(stderr, "For query {} only {} base elements have been found, {} are required.\n", q, results.size(), top_k);
            perror("");
            abort();
        }

        auto wrongIndex = -1;
        auto topList = topListAscending(results);
        auto gt = ground_truth + (q*top_k);
        for (uint32_t k = 0; k < top_k; k++) {
            if(topList[k].getInternalIndex() != gt[k]) {
                wrongIndex = k;
                break;
            }
        }

        if(wrongIndex != -1) {
            for (uint32_t k = 0; k < top_k; k++) 
                fmt::print("{:4}: {:8} ?= {:8} -> {} \n", k, gt[k], topList[k].getInternalIndex(), topList[k].getDistance());
            fmt::print(stderr, "Found entry in gt list which is not the same as computed gt at pos {}\n", wrongIndex);
            perror("");
            abort();
        }
    }
}

/**
 * Compute the gt data
 */
static std::vector<uint32_t> compute_gt(const deglib::StaticFeatureRepository& base_repo, const deglib::StaticFeatureRepository& query_repo, const uint32_t k_target) {
    const auto base_size = (uint32_t)base_repo.size();
    const auto query_size = (uint32_t)query_repo.size();
    const auto dims = base_repo.dims();

    const deglib::Metric metric = deglib::Metric::L2;
    const auto feature_space = deglib::FloatSpace(dims, metric);
    const auto dist_func = feature_space.get_dist_func();
    const auto dist_func_param = feature_space.get_dist_func_param();

    auto count = 0;
    auto topLists = std::vector<uint32_t>(k_target*query_size);
    #pragma omp parallel for
    for (int q = 0; q < (int)query_size; q++) {
        const auto query = query_repo.getFeature(q);

        auto worst_distance = std::numeric_limits<float>::max();
        auto results = deglib::search::ResultSet(); 
        for (uint32_t b = 0; b < base_size; b++) {
            const auto distance = dist_func(query, base_repo.getFeature(b), dist_func_param);
            if(distance < worst_distance) {
                results.emplace(b, distance);
                if (results.size() > k_target) {
                    results.pop();
                    worst_distance = results.top().getDistance();
                }
            }
        }

        if (results.size() != k_target) {
            fmt::print(stderr, "For query {} only {} base elements have been found, {} are required.\n", q, results.size(), k_target);
            perror("");
            abort();
        }

        auto topList = topLists.data() + (k_target*q);
        for(int32_t i = k_target - 1; i >= 0; i--) {
            topList[i] = results.top().getInternalIndex();
            results.pop();
        }

        #pragma omp critical
        {
            count++;
            if(count % 100 == 0)
                fmt::print("Computed {} ground truth lists\n", count);
        }
    }

    return topLists;
}

/**
 * Subsample a repository. Return the selected indices
 */
static std::vector<uint32_t> subsample(const deglib::StaticFeatureRepository& repository, const uint32_t count) {
    const auto dims = repository.dims();
    const auto size = repository.size();
    const auto step_size = (uint32_t) size / count;
    auto output = std::vector<uint32_t>(count);
    for (uint32_t i = 0; i < count; i++) 
        output[i] = i * step_size;
    return output;
}

/**
 * Collect the features for the given indices. The result vector contains all feature values of all indicies.
 */
static std::vector<float> collect(const deglib::StaticFeatureRepository& repository, const std::vector<uint32_t> indices) {
    const auto dims = repository.dims();
    const auto size = indices.size();
    auto output = std::vector<float>(dims*size);
    for (uint32_t i = 0; i < size; i++) {
        const auto feature = repository.getFeature(indices[i]);
        std::memcpy(output.data() + (i*dims), feature, sizeof(float) * dims);
    }
    return output;
}

/**
 * Create a list of random indices for the given repository
 */
static std::vector<uint32_t> createRandomOrder(const deglib::StaticFeatureRepository& repository) {
    const auto size = repository.size();
    auto output = std::vector<uint32_t>(size);
    for (uint32_t i = 0; i < size; i++) 
        output[i] = i;
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(output), std::end(output), rng);
    return output;
}

/**
 * Rewrite the gt file
 */
static void reorderEntryAndGroundTruthFile(const char *file, const std::vector<uint32_t>& order) {
    size_t top_k;
    size_t gt_size;
    const auto ground_truth_f = deglib::fvecs_read(file, top_k, gt_size);
    const auto ground_truth = (uint32_t*)ground_truth_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)

    const auto size = top_k*gt_size;
    for (size_t i = 0; i < size; i++) 
        ground_truth[i] = order[ground_truth[i]];

    ivecs_write(file, top_k, gt_size, ground_truth);
}

/**
 * Rewrite the feature file
 */
static void reorderFeatureFile(const deglib::StaticFeatureRepository& repository, const char *file, const std::vector<uint32_t>& order) {

    const auto size = repository.size();
    const auto dims = repository.dims();
    auto fvs = std::make_unique<float[]>(size * dims);
    for (size_t old_index = 0; old_index < size; old_index++) {
        auto new_index = order[old_index];
        auto fv = fvs.get() + (new_index * dims);
        std::memcpy(fv, repository.getFeature(old_index), sizeof(float) * dims);
    }
    fvecs_write(file, dims, size, fvs.get());
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

    #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(8); // Use 8 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
    #endif

    const auto data_path = std::filesystem::path(DATA_PATH);

    // ---------------------------- SIFT1M ------------------------------
    // const auto base_file                 = (data_path / "SIFT1M/sift_base.fvecs").string();
    // const auto query_file                = (data_path / "SIFT1M/sift_query.fvecs").string();
    // const auto gt_file                   = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    // const auto explore_feature_file      = (data_path / "SIFT1M/sift_explore_query1.fvecs").string();
    // const auto explore_entry_vertex_file = (data_path / "SIFT1M/sift_explore_entry_vertex1.ivecs").string();
    // const auto explore_ground_truth_file = (data_path / "SIFT1M/sift_explore_ground_truth1.ivecs").string();

    // ---------------------------- ImageNet1k ------------------------------
    // const auto reorder_file              = (data_path / "ImageNet1kRand/reorder.ivecs").string();
    // const auto base_file                 = (data_path / "ImageNet1kRand/clip_base.fvecs").string();
    // const auto query_file                = (data_path / "ImageNet1kRand/clip_query.fvecs").string();
    // const auto gt_file                   = (data_path / "ImageNet1kRand/clip_groundtruth.ivecs").string();
    // const auto explore_feature_file      = (data_path / "ImageNet1kRand/clip_explore_query.fvecs").string();
    // const auto explore_entry_vertex_file = (data_path / "ImageNet1kRand/clip_explore_entry_vertex.ivecs").string();
    // const auto explore_ground_truth_file = (data_path / "ImageNet1kRand/clip_explore_ground_truth.ivecs").string();

    // ---------------------------- unsplash ------------------------------
    // const auto base_file                 = (data_path / "unsplash/unsplash_clip_base.fvecs").string();
    // const auto query_file                = (data_path / "unsplash/unsplash_clip_query.fvecs").string();
    // const auto gt_file                   = (data_path / "unsplash/unsplash_clip_groundtruth.ivecs").string();
    // const auto explore_feature_file      = (data_path / "unsplash/unsplash_clip_explore_query.fvecs").string();
    // const auto explore_entry_vertex_file = (data_path / "unsplash/unsplash_clip_explore_entry_vertex.ivecs").string();
    // const auto explore_ground_truth_file = (data_path / "unsplash/unsplash_clip_explore_ground_truth.ivecs").string();

    // ---------------------------- unsplash1M ------------------------------
    // const auto base_file                 = (data_path / "unsplash1m" / "unsplash1m_clip_base.fvecs").string();
    // const auto query_file                = (data_path / "unsplash1m" / "unsplash1m_clip_query.fvecs").string();
    // const auto gt_file                   = (data_path / "unsplash1m" / "unsplash1m_clip_groundtruth.ivecs").string();
    // const auto explore_feature_file      = (data_path / "unsplash1m" / "unsplash1m_clip_explore_query.fvecs").string();
    // const auto explore_entry_vertex_file = (data_path / "unsplash1m" / "unsplash1m_clip_explore_entry_vertex.ivecs").string();
    // const auto explore_ground_truth_file = (data_path / "unsplash1m" / "unsplash1m_clip_explore_ground_truth.ivecs").string();

    // ---------------------------- unsplash uint8 ------------------------------
    // const auto base_file                 = (data_path / "unsplash_uint8/unsplash_clip_uint8_base.fvecs").string();
    // const auto query_file                = (data_path / "unsplash_uint8/unsplash_clip_uint8_query.fvecs").string();
    // const auto gt_file                   = (data_path / "unsplash_uint8/unsplash_clip_uint8_groundtruth.ivecs").string();
    // const auto explore_feature_file      = (data_path / "unsplash_uint8/unsplash_clip_uint8_explore_query.fvecs").string();
    // const auto explore_entry_vertex_file = (data_path / "unsplash_uint8/unsplash_clip_uint8_explore_entry_vertex.ivecs").string();
    // const auto explore_ground_truth_file = (data_path / "unsplash_uint8/unsplash_clip_uint8_explore_ground_truth.ivecs").string();


    // // ---------------------------- Deep1M ------------------------------
    const auto base_file                 = (data_path / "deep1m/deep1m_base.fvecs").string();
    const auto query_file                = (data_path / "deep1m/deep1m_query.fvecs").string();
    const auto gt_file                   = (data_path / "deep1m/deep1m_groundtruth.ivecs").string();
    const auto explore_feature_file      = (data_path / "deep1m/deep1m_explore_query.fvecs").string();
    const auto explore_entry_vertex_file = (data_path / "deep1m/deep1m_explore_entry_vertex.ivecs").string();
    const auto explore_ground_truth_file = (data_path / "deep1m/deep1m_explore_ground_truth.ivecs").string();


    // ---------------------------- Deep10M ------------------------------
    // const auto base_file                 = (data_path / "deep10m/deep10m_base.fvecs").string();
    // const auto query_file                = (data_path / "deep10m/deep10m_query.fvecs").string();
    // const auto gt_file                   = (data_path / "deep10m/deep10m_groundtruth.ivecs").string();
    // const auto explore_feature_file      = (data_path / "deep10m/deep10m_explore_query.fvecs").string();
    // const auto explore_entry_vertex_file = (data_path / "deep10m/deep10m_explore_entry_vertex.ivecs").string();
    // const auto explore_ground_truth_file = (data_path / "deep10m/deep10m_explore_ground_truth.ivecs").string();


    const uint32_t k_target = 100;
    const uint32_t exploration_count = 10000;
    const uint32_t exploration_k = 1000;
    const auto base_repository = deglib::load_static_repository(base_file.c_str());

    // compute ground truth
    // {
    //     const auto query_repository = deglib::load_static_repository(query_file.c_str());
    //     // check_gt(base_repository, query_repository, gt_file.c_str());

    //     const auto start = std::chrono::system_clock::now();
    //     const auto topLists = compute_gt(base_repository, query_repository, k_target);
    //     auto duration = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count());
    //     fmt::print("Computing {:5} top lists of a {:8} base took {:5}s \n", query_repository.size(), base_repository.size(), duration);

    //     // store ground truth
    //     ivecs_write(gt_file.c_str(), k_target, query_repository.size(), topLists.data());
    // }

    // create exploration data
    {
        const auto exploration_entry_ids = subsample(base_repository, exploration_count);
        ivecs_write(explore_entry_vertex_file.c_str(), 1, exploration_count, exploration_entry_ids.data());
        fmt::print("sampled {} exploration ids\n", exploration_count);

        const auto exploration_features = collect(base_repository, exploration_entry_ids);
        fvecs_write(explore_feature_file.c_str(), base_repository.dims(), exploration_count, exploration_features.data());
        fmt::print("collected exploration feature with {}x{} dims = {}\n", base_repository.dims(), exploration_count, exploration_features.size());

        const auto explore_repository = deglib::load_static_repository(explore_feature_file.c_str());
        const auto topLists = compute_gt(base_repository, explore_repository, exploration_k);
        ivecs_write(explore_ground_truth_file.c_str(), exploration_k, explore_repository.size(), topLists.data());
        fmt::print("computed exploration gt with {}x{} elements = {}\n", exploration_k, explore_repository.size(), topLists.size());
    }

    // reorder
    // if(std::filesystem::exists(reorder_file.c_str()) == false) {
    //     const auto reorder = createRandomOrder(base_repository);
    //     ivecs_write(reorder_file.c_str(), 1, reorder.size(), reorder.data());
    //     reorderFeatureFile(base_repository, base_file.c_str(), reorder);
    //     reorderEntryAndGroundTruthFile(gt_file.c_str(), reorder);
    //     reorderEntryAndGroundTruthFile(explore_entry_vertex_file.c_str(), reorder);
    //     reorderEntryAndGroundTruthFile(explore_ground_truth_file.c_str(), reorder);
    // }

    fmt::print("Test OK\n");
    return 0;
}