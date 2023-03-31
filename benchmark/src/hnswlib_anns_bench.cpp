#include <omp.h>
#include <fmt/core.h>

#include <algorithm>
#include <unordered_set>

#include "hnswlib.h"
#include "hnsw/utils.h"
#include "stopwatch.h"

static std::vector<std::unordered_set<size_t>> get_ground_truth(const uint32_t* ground_truth, const size_t ground_truth_size, const size_t ground_truth_dims, const size_t k)
{
    // does the ground truth data provide enough top elements the check for k elements?
    if(ground_truth_dims < k) {
        fmt::print(stderr, "Ground truth data has only {} elements but need {}\n", ground_truth_dims, k);
        abort();
    }

    auto answers = std::vector<std::unordered_set<size_t>>(ground_truth_size);
    for (uint32_t i = 0; i < ground_truth_size; i++)
    {
        auto& gt = answers[i];
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) 
            gt.insert(ground_truth[ground_truth_dims * i + j]);
    }

    return answers;
}

static float test_approx(const hnswlib::HierarchicalNSW<float>& appr_alg, const float* query_repository,
                         const std::vector<std::unordered_set<size_t>>& ground_truth, const size_t query_dims,
                         const size_t k)
{
    size_t correct = 0;
    size_t total = 0;
    for (int i = 0; i < ground_truth.size(); i++)
    {
        auto gt = ground_truth[i];
        auto query = query_repository + query_dims * i;
        auto result_queue = appr_alg.searchKnn(query, k);

        total += gt.size();
        while (result_queue.empty() == false)
        {
            if (gt.find(result_queue.top().second) != gt.end()) correct++;
            result_queue.pop();
        }
    }

    return 1.0f * correct / total;
}

static void test_vs_recall(hnswlib::HierarchicalNSW<float>& appr_alg, const float* query_repository,
                           const std::vector<std::unordered_set<size_t>>& ground_truth, const size_t query_dims,
                           const size_t k)
{
    std::vector<size_t> efs = { 100, 140, 171, 206, 249, 500, 1000 };                           // sift1m + UQ-V + crawl
    // std::vector<size_t> efs = { 20, 30, 40, 50, 70, 100, 150, 300 };                           // audio + enron 
    //std::vector<size_t> efs = { 1000, 1500, 2000, 2500, 5000, 10000, 20000, 40000, 80000 };   // GloVe
    
    for (size_t ef : efs)
    {
        appr_alg.setEf(ef);
        appr_alg.metric_distance_computations = 0;
        appr_alg.metric_hops = 0;

        auto stopw = StopW();
        auto recall = test_approx(appr_alg, query_repository, ground_truth, query_dims, k);
        auto time_us_per_query = stopw.getElapsedTimeMicro() / ground_truth.size();

        float distance_comp_per_query = appr_alg.metric_distance_computations / (1.0f * ground_truth.size());
        float hops_per_query = appr_alg.metric_hops / (1.0f * ground_truth.size());

        fmt::print("ef {} \t recall {} \t time_us_per_query {}us, avg distance computations {}, avg hops {}\n", ef,
                   recall, time_us_per_query, distance_comp_per_query, hops_per_query);
        if (recall > 1.0)
        {
            fmt::print("recall {} \t time_us_per_query {}us\n", recall, time_us_per_query);
            break;
        }
    }
}

int main()
{   
    #if defined(__AVX__)
      std::cout << "use AVX2  ..." << std::endl;
    #elif defined(__SSE2__)
      std::cout << "use SSE  ..." << std::endl;
    #else
      std::cout << "use arch  ..." << std::endl;
    #endif

    #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
    #endif

    const int threads = 1;
    const auto data_path = std::filesystem::path(DATA_PATH);
    fmt::print("Data dir  {} \n", data_path.string().c_str());


    
    

    // ----------------------------- 2D Graph ------------------------------------
    // size_t vecdim = 2;
    // const size_t k = 100;  // k at test time

    // const int efConstruction = 500;
    // const int M = 4;
    // const auto path_query = (data_path / "query.fvecs").string();
    // const auto path_groundtruth = (data_path / "query_gt.ivecs").string();
    // const auto path_basedata = (data_path / "base.fvecs").string();
    
    // // ------------------------------------ GloVe ------------------------------------
    // size_t vecdim = 100;
    // const size_t k = 100;  // k at test time

    // // const int efConstruction = 2500;     // HNSW default
    // // const int M = 25;                    // HNSW default
    // // const int maxM0 = M * 2;             // HNSW default
    // // const int seed = 100;                // HNSW default

    // const int efConstruction = 700; // WEAVESS parameter = 700
    // const int M = 50;               // WEAVESS parameter
    // const int maxM0 = 60;           // WEAVESS parameter
    // const int seed = 100;           // HNSW default

    // const auto path_query       = (data_path / "glove-100/glove-100_query.fvecs").string();
    // const auto path_groundtruth = (data_path / "glove-100/glove-100_groundtruth.ivecs").string();
    // const auto path_basedata    = (data_path / "glove-100/glove-100_base.fvecs").string();



    // // ------------------------------------ SIFT ------------------------------------
    // size_t vecdim = 128;
    // const size_t k = 100;  // k at test time

    // // const int efConstruction = 200; // HNSW default
    // // const int M = 16;               // HNSW default
    // // const int maxM0 = M * 2;        // HNSW default
    // // const int seed = 100;           // HNSW default

    // // const int efConstruction = 600; // SSG parameter
    // // const int M = 25;               // SSG parameter
    // // const int maxM0 = M * 2;        // HNSW default
    // // const int seed = 100;           // HNSW default

    // const int efConstruction = 800; // WEAVESS parameter
    // const int M = 40;               // WEAVESS parameter
    // const int maxM0 = 50;           // WEAVESS parameter
    // const int seed = 100;           // HNSW default

    // const auto path_query = (data_path / "SIFT1M/sift_query.fvecs").string();
    // const auto path_groundtruth = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    // const auto path_basedata = (data_path / "SIFT1M/sift_base.fvecs").string();
    // // const auto order_file = (data_path / "SIFT1M/sift_base_order232076720.int").string();
    // const auto order_file = (data_path / "SIFT1M/sift_base_initial_order.int").string();



    // // ------------------------------------ UQ-V ------------------------------------
    // size_t vecdim = 256;
    // const size_t k = 100;  // k at test time

    // const int efConstruction = 200; // WEAVESS parameter
    // const int M = 10;               // WEAVESS parameter
    // const int maxM0 = 40;           // WEAVESS parameter
    // const int seed = 100;           // HNSW default

    // const auto path_query       = (data_path / "uqv" / "uqv_query.fvecs").string();
    // const auto path_groundtruth = (data_path / "uqv" / "uqv_groundtruth.ivecs").string();
    // const auto path_basedata    = (data_path / "uqv" / "uqv_base.fvecs").string();



    // ------------------------------------ enron ------------------------------------
    // size_t vecdim = 1369;
    // const size_t k = 20;  // k at test time

    // const int efConstruction = 900; // WEAVESS parameter
    // const int M = 50;               // WEAVESS parameter
    // const int maxM0 = 80;           // WEAVESS parameter
    // const int seed = 100;           // HNSW default

    // const auto path_query       = (data_path / "enron" / "enron_query.fvecs").string();
    // const auto path_groundtruth = (data_path / "enron" / "enron_groundtruth.ivecs").string();
    // const auto path_basedata    = (data_path / "enron" / "enron_base.fvecs").string();



    // // ------------------------------------ Crawl ------------------------------------
    // size_t vecdim = 300;
    // const size_t k = 100;  // k at test time

    // const int efConstruction = 400; // WEAVESS parameter
    // const int M = 40;               // WEAVESS parameter
    // const int maxM0 = 70;           // WEAVESS parameter
    // const int seed = 100;           // HNSW default

    // const auto path_query       = (data_path / "crawl" / "crawl_query.fvecs").string();
    // const auto path_groundtruth = (data_path / "crawl" / "crawl_groundtruth.ivecs").string();
    // const auto path_basedata    = (data_path / "crawl" / "crawl_base.fvecs").string();



    // // ------------------------------------ audio ------------------------------------
    // size_t vecdim = 192;
    // const size_t k = 20;  // k at test time

    // const int efConstruction = 700; // WEAVESS parameter
    // const int M = 10;               // WEAVESS parameter
    // const int maxM0 = 50;           // WEAVESS parameter
    // const int seed = 100;           // HNSW default

    // const auto path_query       = (data_path / "audio" / "audio_query.fvecs").string();
    // const auto path_groundtruth = (data_path / "audio" / "audio_groundtruth.ivecs").string();
    // const auto path_basedata    = (data_path / "audio" / "audio_base.fvecs").string();



    // // ------------------------------------ Pixabay clipfv ------------------------------------
    // size_t vecdim = 768;
    // const size_t k = 100;  // k at test time

    // const int efConstruction = 800; // WEAVESS parameter
    // const int M = 40;               // WEAVESS parameter
    // const int maxM0 = 50;           // WEAVESS parameter
    // const int seed = 100;           // HNSW default

    // const auto path_query       = (data_path / "pixabay/pixabay_clipfv_query.fvecs").string();
    // const auto path_groundtruth = (data_path / "pixabay/pixabay_clipfv_groundtruth.ivecs").string();
    // const auto path_basedata    = (data_path / "pixabay/pixabay_clipfv_base.fvecs").string();

    // ------------------------------------ Pixabay gpret ------------------------------------
    size_t vecdim = 1024;
    const size_t k = 100;  // k at test time

    const int efConstruction = 800; // WEAVESS parameter
    const int M = 40;               // WEAVESS parameter
    const int maxM0 = 50;           // WEAVESS parameter
    const int seed = 100;           // HNSW default

    const auto path_query       = (data_path / "pixabay/pixabay_gpret_query.fvecs").string();
    const auto path_groundtruth = (data_path / "pixabay/pixabay_gpret_groundtruth.ivecs").string();
    const auto path_basedata    = (data_path / "pixabay/pixabay_gpret_base.fvecs").string();

    // ------------------------------------------------------------------------------
    // ------------------------------------ HNSW ------------------------------------
    // ------------------------------------------------------------------------------
    char path_index[1024];
    {
        const auto path_index_template = (data_path / "hnsw" / "ef_%d_M_%d_maxM0_%d.hnsw").string();
        std::sprintf(path_index, path_index_template.c_str(), efConstruction, M, maxM0);
    }


    hnswlib::L2Space l2space(vecdim);
    hnswlib::HierarchicalNSW<float>* appr_alg;
    if (exists_test(path_index))
    {
        std::cout << "Loading index from " << path_index << ":" << std::endl;
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, path_index, false);
        std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading index" << std::endl;
    }
    else
    {
         
        std::cout << "Load Data" << std::endl;
        size_t base_size;
        auto base_features = fvecs_read(path_basedata.c_str(), vecdim, base_size);
        std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading data" << std::endl;


        // the order in which the features should be used
        // std::error_code ec{};
        // auto ifstream = std::ifstream(order_file.c_str(), std::ios::binary);
        // auto order_array = std::make_unique<uint32_t[]>(base_size);
        // ifstream.read(reinterpret_cast<char*>(order_array.get()), base_size * sizeof(uint32_t));
        // ifstream.close();

        std::cout << "Creating index:" << std::endl;
        appr_alg = new hnswlib::HierarchicalNSW<float>(&l2space, base_size, M, efConstruction, seed, maxM0);
        std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after creating index" << std::endl;

        std::cout << "Building index:" << std::endl;
        // add first data as a root node
        {
            auto label = 0;
            // auto label = order_array[0];
            auto feature = base_features + vecdim * label;
            appr_alg->addPoint((void*)(feature), (size_t)label);
        }

        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 10000;

        omp_set_num_threads(threads);
#pragma omp parallel for
        for (int i = 1; i < base_size; i++)
        {
            int label = 0;

#pragma omp critical
            {
                label = i;
                // label = order_array[i];
                if (i % report_every == 0)
                {
                    std::cout << i / (0.01 * base_size) << " %, "
                              << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips "
                              << 1e-6 * stopw_full.getElapsedTimeMicro() << "s "
                              << " Mem: " << getCurrentRSS() / 1000000 << " Mb" << std::endl;
                    stopw.reset();
                }
            }

            auto feature = base_features + vecdim * label;
            appr_alg->addPoint((void*)(feature), (size_t)label);
        }
        std::cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds. Mem: " << getCurrentRSS() / 1000000 << " Mb. Peak Mem: " << getPeakRSS() / 1000000 << " Mb." << std::endl;

        std::cout << "Store graph " << path_index << std::endl;
        appr_alg->saveIndex(path_index);
    }


    std::cout << "Load Query Data" << std::endl;
    size_t top_k;
    size_t query_size;
    auto ground_truth = ivecs_read(path_groundtruth.c_str(), top_k, query_size);
    auto query_features = fvecs_read(path_query.c_str(), vecdim, query_size);
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading " << query_size << " query data with TOP" << top_k << " ground truth information " << std::endl;

    // test ground truth
    fmt::print("Parsing gt:\n");
    auto answer = get_ground_truth(ground_truth, query_size, top_k, k);
    fmt::print("Loaded gt with k={}\n", k);
    test_vs_recall(*appr_alg, query_features, answer, vecdim, k);
    fmt::print("Actual memory usage: {} Mb\n", getCurrentRSS() / 1000000);
    fmt::print("Test ok\n");

    return 0;
}