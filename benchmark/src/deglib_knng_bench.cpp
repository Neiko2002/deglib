#include <fmt/core.h>

#include <algorithm>
#include <random>
#include <math.h>
#include <limits>

#include "benchmark.h"
#include "deglib.h"

// smaller graphs with low k values should use higher eps values
static const tsl::robin_map<uint32_t, float> deg_k_to_eps = {{4, 1.5f}, {6, 1.3f}, {8, 1.3f}, {10, 1.3f}, {12, 0.9f}, {14, 0.8f}, {16, 0.7f}, {18, 0.6f}};

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
 * Convert a numpy file generated by faiss to a top list file
 */
static void store_top_list(const char* numpy_file, const char* top_list_file, const uint32_t size) {

    std::error_code ec{};
    auto file_size = std::filesystem::file_size(numpy_file, ec);
    if (ec != std::error_code{})
    {
        fmt::print(stderr, "error when accessing top list file, size is: {} message: {} \n", numpy_file, file_size, ec.message());
        perror("");
        abort();
    }

    auto in = std::ifstream(numpy_file, std::ios::binary);
    if (!in.is_open())
    {
        fmt::print(stderr, "could not open {}\n", numpy_file);
        perror("");
        abort();
    }

    // check open file for write
    fmt::print("Storing top lists to {}\n", top_list_file);
    auto out = std::ofstream(top_list_file, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        fmt::print(stderr, "Error in open file {}\n", top_list_file);
        perror("");
        abort();
    }

    // dims per vector
    uint32_t dims = (uint32_t) ((file_size / size) / 4); // expecting uint32_t
    fmt::print("Read numpy files with {} vectors of size {}\n", size, dims);
    auto topList = std::vector<uint32_t>();
    topList.reserve(dims);

    // read all the 
    for (size_t i = 0; i < size; i++) {
        in.read(reinterpret_cast<char*>(topList.data()), dims * sizeof(uint32_t));
        topList[0] = dims - 1;  // replace the self reference with the size of the vector
        out.write(reinterpret_cast<const char*>(topList.data()), dims * sizeof(uint32_t));    
        topList.clear();
    }
}

static void create_explore_ground_truth(const deglib::FeatureRepository& repository, const char* top_list_file, const char* feature_file, const char* ground_truth_file, const char* entry_node_file, const uint32_t step_size) {
    fmt::print("Build explore ground truth with {} elements \n", repository.size() / step_size);

    size_t top_list_dims;
    size_t top_list_count;
    const auto top_list_f = deglib::fvecs_read(top_list_file, top_list_dims, top_list_count);
    const auto top_list = (uint32_t*)top_list_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("Load TopList data with {} elements and k={}\n", top_list_count, top_list_dims);

    // create feature file
    const auto size = (uint32_t) repository.size();
    {
        fmt::print("Storing explore ground truth features to {}\n", feature_file);
        auto out = std::ofstream(feature_file, std::ios::out | std::ios::binary);

        // check open file for write
        if (!out.is_open()) {
            fmt::print(stderr, "Error in open file {}\n", feature_file);
            perror("");
            abort();
        }

        const auto feature_dim = (uint32_t) repository.dims();
        for (uint32_t i = 0; i < size; i+=step_size) {
            const auto feature = repository.getFeature(i);
            out.write(reinterpret_cast<const char*>(&feature_dim), sizeof(feature_dim));    
            out.write(reinterpret_cast<const char*>(feature), sizeof(float) * feature_dim);    
        }

        out.close();
    }
    
    // create ground truth top list
    {
        fmt::print("Storing explore ground truth top list to {}\n", ground_truth_file);
        auto out = std::ofstream(ground_truth_file, std::ios::out | std::ios::binary);

        // check open file for write
        if (!out.is_open()) {
            fmt::print(stderr, "Error in open file {}\n", ground_truth_file);
            perror("");
            abort();
        }

        const auto ground_truth_dims = (uint32_t) top_list_dims;
        for (uint32_t i = 0; i < size; i+=step_size) {
            const auto top_list_entry =  top_list + i * ground_truth_dims;
            out.write(reinterpret_cast<const char*>(&ground_truth_dims), sizeof(ground_truth_dims));    
            out.write(reinterpret_cast<const char*>(top_list_entry), sizeof(uint32_t) * ground_truth_dims);    
        }

        out.close();
    }

    // create entry node
    {
        fmt::print("Storing explore entry node to {}\n", entry_node_file);
        auto out = std::ofstream(entry_node_file, std::ios::out | std::ios::binary);

        // check open file for write
        if (!out.is_open()) {
            fmt::print(stderr, "Error in open file {}\n", entry_node_file);
            perror("");
            abort();
        }

        const auto entry_node_size = (uint32_t) 1;
        for (uint32_t i = 0; i < size; i+=step_size) {
            out.write(reinterpret_cast<const char*>(&entry_node_size), sizeof(entry_node_size));    
            out.write(reinterpret_cast<const char*>(&i), sizeof(i));    
        }

        out.close();
    }
}

/**
 * Compute a new top list with the help of a graph
 * Note: this is not a perfect top list
 */
static void store_top_list(const deglib::search::SearchGraph& graph, const deglib::FeatureRepository& repository, const char* top_list_file, const uint32_t k, const float eps) {
    fmt::print("Build top lists for repo with {} elements \n", repository.size());

    // reproduceable entry point for the graph search
    const uint32_t entry_node_id = 0;
    const auto entry_node_indices = std::vector<uint32_t> { graph.getInternalIndex(entry_node_id) };

    uint32_t size = (uint32_t)repository.size();
    auto topList = std::vector<uint32_t>();
    topList.reserve(size * (k + 1)); // size of vector + neighbor ids vector
    for (uint32_t i = 0; i < repository.size(); i++)
    {
        auto query = reinterpret_cast<const std::byte*>(repository.getFeature(i));
        auto result_queue = graph.search(entry_node_indices, query, eps, (k + 1)); // +1 for self reference
        auto sorted_result = topListAscending(result_queue);

        topList.push_back(k);                                   // size of the vector
        for(uint32_t r = 1; r < sorted_result.size(); r++) {    // ignore the self reference in the result list
            const auto internal_index = sorted_result[r].getInternalIndex();
            const auto external_id = graph.getExternalLabel(internal_index);
            topList.push_back(external_id);
        }

        if(i % 10000 == 0)
           fmt::print("Processed {} elements \n", i);
    }


    fmt::print("Storing top lists to {}\n", top_list_file);
    auto out = std::ofstream(top_list_file, std::ios::out | std::ios::binary);

    // check open file for write
    if (!out.is_open()) {
        fmt::print(stderr, "Error in open file {}\n", top_list_file);
        perror("");
        abort();
    }

    // store all top lists in the same order than the repository
    uint64_t byte_size = topList.size() * sizeof(uint32_t);
    out.write(reinterpret_cast<const char*>(topList.data()), byte_size);    
    out.close();

    fmt::print("Finish building and storing top lists\n\n");
}

static bool create_knng(const deglib::FeatureRepository& repository, const char* top_list_file, const std::filesystem::path& knng_dir, const uint8_t k_target) {
    fmt::print("Build and store KNNGs with perfect egdes\n");

    size_t top_list_dims;
    size_t top_list_count;
    const auto top_list_f = deglib::fvecs_read(top_list_file, top_list_dims, top_list_count);
    const auto top_list = (uint32_t*)top_list_f.get(); // not very clean, works as long as sizeof(int) == sizeof(float)
    fmt::print("Load TopList data with {} elements and k={}\n", top_list_count, top_list_dims);

    if(top_list_count != repository.size()) {
        fmt::print(stderr, "The number of elements in the TopList file {} is different than in the repository: {} vs {}\n", top_list_file, top_list_count, repository.size());
        return false;
    }

    if(top_list_dims < k_target) {
        fmt::print(stderr, "k_target = {} is higher than the TopList size = {} of the file {} \n", k_target, top_list_dims, top_list_file);
        return false;
    }

    const auto feature_space = deglib::FloatSpace(128, deglib::Metric::L2);
    const auto dist_func = feature_space.get_dist_func();
    const auto dist_func_param = feature_space.get_dist_func_param();

    // create graphs with different states of randomness
    auto neighbor_ids = std::vector<uint32_t>();
    auto neighbor_weights = std::vector<float>();
    neighbor_ids.reserve(k_target);
    neighbor_weights.reserve(k_target);
    for (uint8_t k = 4; k <= k_target; k+=2) {

        // write graph to file 
        auto filename = fmt::format("knng_{}.deg", k);
        auto path_to_graph = (knng_dir / filename).string();
        auto out = std::ofstream(path_to_graph, std::ios::out | std::ios::binary);

        // check open file for write
        if (!out.is_open()) {
            fmt::print(stderr, "Error in open file {}\n", path_to_graph);
            return false;
        }

        // store feature space information
        uint8_t data_type = 1; // 1=float
        out.write(reinterpret_cast<const char*>(&data_type), sizeof(data_type));
        uint16_t feature_dim = uint16_t(repository.dims());
        out.write(reinterpret_cast<const char*>(&feature_dim), sizeof(feature_dim));

        // store graph information
        uint32_t size = uint32_t(top_list_count);
        out.write(reinterpret_cast<const char*>(&size), sizeof(size));
        uint8_t edges_per_node = k;
        out.write(reinterpret_cast<const char*>(&edges_per_node), sizeof(edges_per_node));

        // store the node and edge information
        for (uint32_t i = 0; i < size; i++) {
            const auto fv1 = repository.getFeature(i);

            // copy some perfect ids from the top list
            neighbor_ids.clear();
            for (uint32_t pos = 0; pos < k; pos++) 
                neighbor_ids.emplace_back(top_list[i * top_list_dims + pos]);
            
            // sort ids since its required by the graph
            std::sort(neighbor_ids.begin(), neighbor_ids.end());

            // compute the weights
            neighbor_weights.clear();
            for (uint32_t pos = 0; pos < k; pos++) {
                const auto fv2 = repository.getFeature(neighbor_ids[pos]);
                const auto dist = dist_func(fv1, fv2, dist_func_param);
                neighbor_weights.emplace_back(dist);
            }

            // store everything to file
            out.write(reinterpret_cast<const char*>(fv1), feature_dim * 4); 
            out.write(reinterpret_cast<const char*>(neighbor_ids.data()), k * 4);       
            out.write(reinterpret_cast<const char*>(neighbor_weights.data()), k * 4); 
            out.write(reinterpret_cast<const char*>(&i), sizeof(i));
        }

        fmt::print("Write graph {} \n\n", filename);
    }

    fmt::print("Finish building and storing knng graphs\n\n");
    return true;
}

/**
 * The DEG will not be improved just created with our build algorithm.
 */ 
static bool create_deg_add_only(const deglib::FeatureRepository& repository, const std::filesystem::path& deg_dir, const uint8_t k_target) {
    fmt::print("Build and store DEG add only graphs\n");

    const uint32_t max_node_count = uint32_t(repository.size());
    const auto dims = repository.dims();

    // create graphs with different states of randomness
    for (uint8_t k = 24; k <= k_target; k+=10) {
        fmt::print("Build DEG {}\n", k);
        auto eps = 0.2f;

        // some k values have special eps
        auto it = deg_k_to_eps.find(k);
        if(it != deg_k_to_eps.end()) 
            eps = it->second;
        
        // create the graph and its builder object
        auto rnd = std::mt19937(7);
        const auto highLID = true;
        const auto feature_space = deglib::FloatSpace(dims, deglib::Metric::L2);
        auto graph = deglib::graph::SizeBoundedGraph(max_node_count, k, feature_space);
        auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, k, eps, highLID, 0, 0.f, false, 2, 10, 0, 0);

        // provide all features to the graph builder at once. In an online system this will be called 
        for (uint32_t label = 0; label < repository.size(); label++) {
            auto feature = reinterpret_cast<const std::byte*>(repository.getFeature(label));
            auto feature_vector = std::vector<std::byte>{feature, feature + dims * sizeof(float)};
            builder.addEntry(label, feature_vector);
        }

        // check the integrity of the graph during the graph build process
        const auto log_after = 10000;
        const auto start = std::chrono::system_clock::now();
        const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {

            if(status.added % log_after == 0) {
                auto quality = deglib::analysis::calc_avg_edge_weight(graph);
                auto connected = deglib::analysis::check_graph_connectivity(graph);
                auto duration = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count());
                fmt::print("{:7} elements, in {:5}s, quality {:4.2f}, connected {} \n", status.added, duration, quality, connected);
            }

            // check the graph from time to time
            if(status.added % log_after == 0) {
                auto valid = deglib::analysis::check_graph_validation(graph, uint32_t(status.added - status.deleted), true);
                if(valid == false) {
                    builder.stop();
                    fmt::print("Invalid graph, build process is stopped");
                } 
            }
        };

        // start the build process
        builder.build(improvement_callback, false);

        // store the graph
        const auto filename = fmt::format("k{}nns_{}D_L2_AddK{}Eps{}{}.deg", k, dims, k, eps, highLID ? "High" : "Low");
        const auto graph_file = (deg_dir / filename).string();
        graph.saveGraph(graph_file.c_str());
        fmt::print("Write graph {} \n\n", filename);
    }

    fmt::print("Finish building and storing DEG in add only mode\n\n");
    return true;
}

static std::vector<deglib::search::ObjectDistance> fullSearch(const deglib::graph::SizeBoundedGraph& graph, const uint32_t query_index, const uint8_t k_target) {
    const auto graph_size = (uint32_t)graph.size();
    const auto query = graph.getFeatureVector(query_index);

    const auto dist_func = graph.getFeatureSpace().get_dist_func();
    const auto dist_func_param = graph.getFeatureSpace().get_dist_func_param();
        
    auto worst_distance = std::numeric_limits<float>::max();
    auto results = deglib::search::ResultSet(); 
    deglib::MemoryCache::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(0)));
    for (uint32_t i = 0; i < graph_size; i++) {
        deglib::MemoryCache::prefetch(reinterpret_cast<const char*>(graph.getFeatureVector(std::min(i + 1, graph_size - 1))));
        const auto distance = dist_func(query, graph.getFeatureVector(i), dist_func_param);
        if(i != query_index && distance < worst_distance) {
            results.emplace(i, distance);
             if (results.size() > k_target) {
                results.pop();
                worst_distance = results.top().getDistance();
            }
        }
    }
    return topListAscending(results);
}

/**
 * Similar to create_deg_add_only but it will uses a perfect TOP list to connect the best nodes.
 * First all nodes will be added and than the edges. This is not the same as adding nodes and perfect edges at the same time.
 */ 
static bool create_deg_add_only_perfect(const deglib::FeatureRepository& repository, const std::filesystem::path& deg_dir, const uint8_t k_target) {
    fmt::print("Build and store DEG add only graph using a perfect top list\n");

    const auto dims = repository.dims();
    const auto feature_space = deglib::FloatSpace(dims, deglib::Metric::L2);
    const auto dist_func = feature_space.get_dist_func();
    const auto dist_func_param = feature_space.get_dist_func_param();

    // create the graph and add all nodes, without any edges
    const auto start = std::chrono::system_clock::now();
    const uint8_t edges_per_node = k_target;
    const uint32_t node_count = uint32_t(repository.size());
    auto graph = deglib::graph::SizeBoundedGraph(node_count, edges_per_node, feature_space);

    // add the initial nodes (edges_per_node + 1)
    {
        const auto size = (uint32_t)(edges_per_node + 1);
        for (uint32_t y = 0; y < size; y++) {
            const auto query = reinterpret_cast<const std::byte*>(repository.getFeature(y));
            const auto internal_index = graph.addNode(y, query);

            auto neighbor_indices = std::vector<uint32_t>();
            auto neighbor_weights = std::vector<float>();
            for (uint32_t x = 0; x < size; x++) {
                if(x == internal_index) continue;
                neighbor_indices.emplace_back(x);
                neighbor_weights.emplace_back(dist_func(query, reinterpret_cast<const std::byte*>(repository.getFeature(x)), dist_func_param));
            }
            graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());
        }
    }

    // add the remaining nodes
    for (uint32_t label = edges_per_node + 1; label < node_count; label++) {
        const auto internal_index = graph.addNode(label, reinterpret_cast<const std::byte*>(repository.getFeature(label)));
        const auto top_list = fullSearch(graph, internal_index, k_target);

        // remove the worst edge of the good neighbors and connect them with this new node
        auto new_neighbors = std::vector<std::pair<uint32_t, float>>();
        for (int i = 0; i < top_list.size() && new_neighbors.size() < edges_per_node; i++) {
            const auto top_neighbor = top_list[i];

            // check if the node is already in the edge list of the new node (added during a previous loop-run)
            // since all edges are undirected and the edge information of the new node does not yet exist, we search the other way around.
            if(graph.hasEdge(top_neighbor.getInternalIndex(), internal_index)) 
                continue;

            // find the worst edge of the new neighbor
            uint32_t bad_neighbor_index = 0;
            float bad_neighbor_weight = 0;
            const auto neighbor_weights = graph.getNeighborWeights(top_neighbor.getInternalIndex());
            const auto neighbor_indices = graph.getNeighborIndices(top_neighbor.getInternalIndex());
            for (size_t edge_idx = 0; edge_idx < edges_per_node; edge_idx++) {
                const auto neighbor_index = neighbor_indices[edge_idx];
                const auto neighbor_weight = neighbor_weights[edge_idx];

                // the suggest neighbors might already be in the edge list of the new node
                // the weight of the neighbor might not be worst than the current worst one
                if(bad_neighbor_weight < neighbor_weight && graph.hasEdge(neighbor_index, internal_index) == false) {
                    bad_neighbor_index = neighbor_index;
                    bad_neighbor_weight = neighbor_weight;
                }          
            }

            // this should not be possible, otherwise the new node is connected to every node in the neighbor-list of the result-node and still has space for more
            if(bad_neighbor_weight <= 0) {
                fmt::print(stderr, "it was not possible to find a bad edge (best weight {}) in the neighbor list of node {} which would connect to node {} \n", bad_neighbor_weight, top_neighbor.getInternalIndex(), internal_index);
                perror("");
                abort();
            }

            // place the new node in the edge list of the result-node
            graph.changeEdge(top_neighbor.getInternalIndex(), bad_neighbor_index, internal_index, top_neighbor.getDistance());
            new_neighbors.emplace_back(top_neighbor.getInternalIndex(), top_neighbor.getDistance());

            // place the new node in the edge list of the worst edge neighbor
            const auto bad_neighbor_distance = dist_func(graph.getFeatureVector(internal_index), graph.getFeatureVector(bad_neighbor_index), dist_func_param);
            graph.changeEdge(bad_neighbor_index, top_neighbor.getInternalIndex(), internal_index, bad_neighbor_distance);
            new_neighbors.emplace_back(bad_neighbor_index, bad_neighbor_distance);
        }

        // sort the neighbors by their neighbor indices and store them in the new node
        std::sort(new_neighbors.begin(), new_neighbors.end(), [](const auto& x, const auto& y){return x.first < y.first;});
        auto neighbor_indices = std::vector<uint32_t>();
        auto neighbor_weights = std::vector<float>();
        for (auto &&neighbor : new_neighbors) {
            neighbor_indices.emplace_back(neighbor.first);
            neighbor_weights.emplace_back(neighbor.second);
        }
        graph.changeEdges(internal_index, neighbor_indices.data(), neighbor_weights.data());

        if(label % 10000 == 0) {
            auto quality = deglib::analysis::calc_avg_edge_weight(graph);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count());
            fmt::print("{:7} elements, in {:5}s, quality {:4.2f}, connected {} \n", label, duration, quality, connected);
        }
    }

    // store the graph
    const auto filename = fmt::format("k{}nns_128D_L2_AddK{}_with_perfect_edges.deg", k_target, k_target);
    const auto graph_file = (deg_dir / filename).string();
    graph.saveGraph(graph_file.c_str());
    fmt::print("Write graph {} \n\n", filename);
}


static void test_limit_distance_computation(const char* graph_file, const deglib::FeatureRepository& query_repository, const std::vector<tsl::robin_set<uint32_t>>& answer, const uint32_t max_distance_count, const uint32_t k) {

    const auto graph = deglib::graph::load_readonly_graph(graph_file);
    const auto entry_node_indices = std::vector<uint32_t> { graph.getInternalIndex(0) };

    // compute graph distortion
    auto distortion = 0.;
    {
        const auto feature_space = deglib::FloatSpace(128, deglib::Metric::L2);
        const auto dist_func = feature_space.get_dist_func();
        const auto dist_func_param = feature_space.get_dist_func_param();
        const auto edges_per_node = graph.getEdgesPerNode();
        const auto size = graph.size();
        for (uint32_t n = 0; n < size; n++) {
            const auto fv1 = graph.getFeatureVector(n);
            const auto neighborIds = graph.getNeighborIndices(n); 
            for (uint32_t e = 0; e < edges_per_node; e++) {
                const auto fv2 = graph.getFeatureVector(neighborIds[e]);
                const auto dist = dist_func(fv1, fv2, dist_func_param);
                distortion += dist;
            }
        }
        distortion /= uint64_t(size) * edges_per_node;
    }

    float best_precision = 0;
    float best_eps = 0;

    std::vector<float> eps_parameter = { 0.2 };
    //std::vector<float> eps_parameter = { 1.5, 2.0, 2.25, 2.5, 2.75, 3, 3.5, 4 };
    //td::vector<float> eps_parameter = { 0.05, 0.06, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3, 3.5, 4 };
    for (float eps : eps_parameter)
    {

        size_t total = 0;
        size_t correct = 0;
        for (int i = 0; i < query_repository.size(); i++)
        {
            auto query = reinterpret_cast<const std::byte*>(query_repository.getFeature(i));
            auto result_queue = graph.search(entry_node_indices, query, eps, k, max_distance_count);

            const auto gt = answer[i];
            total += result_queue.size();
            while (result_queue.empty() == false)
            {
                const auto internal_index = result_queue.top().getInternalIndex();
                const auto external_id = graph.getExternalLabel(internal_index);
                if (gt.find(external_id) != gt.end()) correct++;
                result_queue.pop();
            }
        }

        const auto precision = ((float)correct) / total;
        //fmt::print("{} with distortion of {:0.1f}, precision of {:0.4f} with eps {}\n", graph_file, distortion, precision, eps);

        if(best_precision < precision) {
            best_precision = precision;
            best_eps = eps;
        } else {
            break;
        }
    }

    fmt::print("{} with avg edge weight {:0.1f}, precision of {:0.4f} using eps {}\n", graph_file, distortion, best_precision, best_eps);
}


static void test_limit_distance_computation_knng(const std::filesystem::path& knng_dir, const uint32_t knng_size_max, const deglib::FeatureRepository& query_repository, const std::vector<tsl::robin_set<uint32_t>>& answer, const uint32_t max_distance_count,  const uint32_t k) {

    for (uint32_t knng_size = 40; knng_size <= knng_size_max; knng_size += 10) {
        const auto graph_file = (knng_dir / fmt::format("knng_{}.deg", knng_size)).string();        
        test_limit_distance_computation(graph_file.c_str(), query_repository, answer, max_distance_count, k);
    }
}

static void test_limit_distance_computation_deg(const std::filesystem::path& deg_dir, const uint32_t deg_size_max, const deglib::FeatureRepository& query_repository, const std::vector<tsl::robin_set<uint32_t>>& answer, const uint32_t max_distance_count,  const uint32_t k) {

    for (uint32_t deg_size = 4; deg_size <= deg_size_max; deg_size+=2) {

        // some k values have special eps
        auto eps_build = 0.2f;
        auto it = deg_k_to_eps.find(deg_size);
        if(it != deg_k_to_eps.end()) 
            eps_build = it->second;

        const auto graph_file = (deg_dir / fmt::format("k{}nns_128D_L2_AddK{}Eps{}.deg", deg_size, deg_size, eps_build)).string();    
        test_limit_distance_computation(graph_file.c_str(), query_repository, answer, max_distance_count, k);
    }
}

static void improve_and_test_deg(const char* initial_graph_file, const std::filesystem::path& deg_dir, const deglib::FeatureRepository& query_repository, const std::vector<tsl::robin_set<uint32_t>>& answer, const uint32_t max_distance_count, const uint32_t k_test) {

    auto rnd = std::mt19937(7);
    auto graph = deglib::graph::load_sizebounded_graph(initial_graph_file);
    auto edges_per_node = graph.getEdgesPerNode();
    auto builder = deglib::builder::EvenRegularGraphBuilder(graph, rnd, edges_per_node, 0.2f, edges_per_node, 0.02f, edges_per_node, 0.02f, 2, edges_per_node/2, 1, 0);

    fmt::print("Improve and test graph {}\n", initial_graph_file);
    test_limit_distance_computation(initial_graph_file, query_repository, answer, max_distance_count, k_test);

    const uint32_t log_after = 100000;
    const auto start = std::chrono::system_clock::now();
    auto last_status = deglib::builder::BuilderStatus{};
    const auto improvement_callback = [&](deglib::builder::BuilderStatus& status) {
        if(status.tries % log_after == 0) {
            auto quality = deglib::analysis::calc_avg_edge_weight(graph);
            auto connected = deglib::analysis::check_graph_connectivity(graph);
            auto duration = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count());
            auto avg_improv = uint32_t((status.improved - last_status.improved) / log_after);
            auto avg_tries = uint32_t((status.tries - last_status.tries) / log_after);
            fmt::print("{:7} elements, in {:5}s, with {:8} / {:8} improvements (avg {:2}/{:3}), quality {:6.2f}, connected {} \n", status.added, duration, status.improved, status.tries, avg_improv, avg_tries, quality, connected);

            // check the graph from time to time
            auto valid = deglib::analysis::check_graph_validation(graph, uint32_t(status.added - status.deleted), true);
            if(valid == false) {
                builder.stop();
                fmt::print("Invalid graph, build process is stopped");
            } 

            // test the graph quality
            const auto graph_file = (deg_dir / fmt::format("deg{}_128D_L2_AddK{}Eps0.2_Improve{}Eps0.02_ImproveExt{}-2StepEps0.02_Path{}_it{}m.deg", edges_per_node, edges_per_node, edges_per_node, edges_per_node, edges_per_node/2, status.tries/1000000)).string();
            graph.saveGraph(graph_file.c_str());
            test_limit_distance_computation(graph_file.c_str(), query_repository, answer, max_distance_count, k_test);

            last_status = status;
        }
    };

    builder.build(improvement_callback, true);
}

static void randomize_and_test_knng(const char* initial_graph_file, const std::filesystem::path& knng_dir, const deglib::FeatureRepository& query_repository, const std::vector<tsl::robin_set<uint32_t>>& answer, const uint32_t max_distance_count, const uint32_t k_test) {

    auto graph = deglib::graph::load_sizebounded_graph(initial_graph_file);
    const auto size = graph.size();
    const auto edges_per_node = graph.getEdgesPerNode();

    const auto feature_space = deglib::FloatSpace(128, deglib::Metric::L2);
    const auto dist_func = feature_space.get_dist_func();
    const auto dist_func_param = feature_space.get_dist_func_param();

    fmt::print("Test: Do additional random edges help in a perfect KNNG\n", initial_graph_file);
    test_limit_distance_computation(initial_graph_file, query_repository, answer, max_distance_count, k_test);

    fmt::print("Find the worst egdes in the current graph {}\n", initial_graph_file);
    auto all_sorted_edges = std::vector<std::vector<std::pair<uint32_t,float>>>(size); // for every node there is a vector of sorted edges
    for (uint32_t n = 0; n < size; n++) {
        auto indices = graph.getNeighborIndices(n);
        auto weights = graph.getNeighborWeights(n);

        auto& sorted_edges = all_sorted_edges[n];
        for (size_t e = 0; e < edges_per_node; e++)
            sorted_edges.emplace_back(indices[e], weights[e]);
        
        // sort edges by weight
        std::sort(sorted_edges.begin(), sorted_edges.end(), [](const auto& x, const auto& y){return x.second < y.second;});
    }

    fmt::print("Replace the worst egdes with random edges to produce highways to other graph regions\n");
    auto replace_count = 1;
    auto rnd = std::mt19937(7);
    const auto distrib = std::uniform_int_distribution<uint32_t>(0, uint32_t(graph.size() - 1));
    for (uint8_t i = 0; i < edges_per_node; i++) {
        for (uint32_t n = 0; n < size; n++, replace_count++) {
            const auto bad_edge_index = all_sorted_edges[n][i].first;

            // find a new neighbor which is not yet connected
            auto rnd_edge_index = (uint32_t) distrib(rnd);
            while(graph.hasEdge(n, rnd_edge_index))
                rnd_edge_index = (uint32_t) distrib(rnd);

            // replace edge a bad edge from n with a random one
            const auto rnd_edge_weight = dist_func(graph.getFeatureVector(n), graph.getFeatureVector(rnd_edge_index), dist_func_param);
            graph.changeEdge(n, bad_edge_index, rnd_edge_index, rnd_edge_weight);
            
            // test the new graph from time to time
            if(replace_count % 1000000 == 0) {
                const auto graph_file = (knng_dir / fmt::format("knng_30_rndEdge{}.deg", replace_count)).string();
                graph.saveGraph(graph_file.c_str());
                test_limit_distance_computation(graph_file.c_str(), query_repository, answer, max_distance_count, k_test);
            }
        }
    }
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

    const auto repeat_test = 3;
    const auto data_path = std::filesystem::path(DATA_PATH);

    // create and store top list of all database elements
    // {
    //     //const auto graph_file = (data_path / "deg" / "k24nns_128D_L2_Path10_Rnd3+3_AddK24Eps0.2_ImproveK24Eps0.02_ImproveExtK36-2StepEps0.02.deg").string();
    //     //const auto graph = deglib::graph::load_readonly_graph(graph_file.c_str());
    //     //
    //     //const auto repository_file = (data_path / "SIFT1M/sift_base.fvecs").string();
    //     //const auto repository = deglib::load_static_repository(repository_file.c_str());
    //     //const auto top_list_file = (data_path / "SIFT1M/sift_base_top200_p0.998.ivecs").string();

    // const auto numpy_file = (data_path / "SIFT1M" / "sift_base_top1001.npd").string();
    // const auto top_list_file = (data_path / "SIFT1M" / "sift_base_top1000.ivecs").string();
    // store_top_list(numpy_file.c_str(), top_list_file.c_str(), 1000000);

    const auto numpy_file = (data_path / "glove-100" / "glove_base_top1001.npd").string();
    const auto top_list_file = (data_path / "glove-100" / "glove_base_top1000.ivecs").string();
    store_top_list(numpy_file.c_str(), top_list_file.c_str(), 1183514);

        
    //     const auto repository_file = (data_path / "SIFT1M/sift_base.fvecs").string();
    //     const auto repository = deglib::load_static_repository(repository_file.c_str());

    //     const auto top_list_file = (data_path / "SIFT1M" / "sift_base_top1000.ivecs").string();
    //     const auto explore_entry_node_file = (data_path / "SIFT1M/sift_explore_entry_node.ivecs").string();
    //     const auto explore_feature_file = (data_path / "SIFT1M/sift_explore_query.fvecs").string();
    //     const auto explore_ground_truth_file = (data_path / "SIFT1M/sift_explore_ground_truth.ivecs").string();
    //     create_explore_ground_truth(repository, top_list_file.c_str(), explore_feature_file.c_str(), explore_ground_truth_file.c_str(), explore_entry_node_file.c_str(),100);
    // }

    // create KNN graph with the help of database top list
    // {
    //     const auto repository_file = (data_path / "SIFT1M/sift_base.fvecs").string();
    //     // const auto repository_file = (data_path / "glove-100/glove-100_base.fvecs").string();
    //     const auto repository = deglib::load_static_repository(repository_file.c_str());
    //     create_deg_add_only(repository, (data_path / "deg"), 150);
    //     //for (uint8_t k = 4; k <= 4; k+=2) 
    //     //    create_deg_add_only_perfect(repository, (data_path / "deg/add_perfect_only"), k);
    //     // const auto top_list_file = (data_path / "SIFT1M" / "sift_base_top1000.ivecs").string();
    //     // create_knng(repository, top_list_file.c_str(), (data_path / "knng"), 200);
    // }

    // // test the KNN graph with limited distance compution numbers
    // {
    //     // limit the amount of distance computations, gives a perfect KNN graph problems to reach to good regions in a graph
    //     // Graphs which have same random edges have better navigation properties
    //     //const uint32_t k = 50;
    //     //const float eps = 0.2f;
    //     //const uint32_t max_distance_count = 2000;

    //     // only when the same graph has a lot of freedom (more distance computations) better results can be found
    //     //const uint32_t k = 50;
    //     //const float eps = 0.2f;
    //     //const uint32_t max_distance_count = 20000;

    //     const uint32_t k = 100;
    //     const uint32_t max_distance_count = 5000;

    //     const auto path_query_repository = (data_path / "SIFT1M/sift_query.fvecs").string();
    //     const auto path_query_groundtruth = (data_path / "SIFT1M/sift_groundtruth.ivecs").string();
    //     const auto query_repository = deglib::load_static_repository(path_query_repository.c_str());
    //     fmt::print("{} Query Features with {} dimensions \n", query_repository.size(), query_repository.dims());

    //     size_t ground_truth_dims;
    //     size_t ground_truth_count;
    //     auto ground_truth_f = deglib::fvecs_read(path_query_groundtruth.c_str(), ground_truth_dims, ground_truth_count);
    //     const auto ground_truth = (uint32_t*) ground_truth_f.release(); // not very clean, works as long as sizeof(int) == sizeof(float)
    //     const auto answer = deglib::benchmark::get_ground_truth(ground_truth, query_repository.size(), (uint32_t)ground_truth_dims, k);
    //     delete ground_truth;
    //     fmt::print("{} ground truth {} dimensions \n", ground_truth_count, ground_truth_dims);

    //     // test_limit_distance_computation_knng((data_path / "knng"), 200, query_repository, answer, max_distance_count, k);  
    //     //test_limit_distance_computation_deg((data_path / "deg/best_distortion_decisions/add_only"), 100, query_repository, answer, max_distance_count, k); 

    //     // const auto graph_file = (data_path / "deg/best_distortion_decisions" / "k30nns_128D_L2_AddK30Eps0.2_ImproveK30Eps0.02_ImproveExtK30-2StepEps0.02_Path20_Rnd15+15.deg").string();
    //     //test_limit_distance_computation(graph_file.c_str(), query_repository, answer, max_distance_count, k);  // DEG file

    //     //const auto graph_file = (data_path / "deg/best_distortion_decisions/improve/deg30_128D_L2_AddK30Eps0.2_Improve30Eps0.02_ImproveExt30-2StepEps0.02_Path15_it80m.deg").string();
    //     //const auto graph_file = (data_path / "deg/best_distortion_decisions/add_only/k30nns_128D_L2_AddK30Eps0.2.deg").string();
    //     // improve_and_test_deg(graph_file.c_str(), (data_path / "deg/best_distortion_decisions/improve1"), query_repository, answer, max_distance_count, k);

    //     const auto graph_file = (data_path / "knng/knng_30.deg").string();
    //     randomize_and_test_knng(graph_file.c_str(), (data_path / "knng/rnd"), query_repository, answer, max_distance_count, k);
    // }

    fmt::print("Test OK\n");
    return 0;
}