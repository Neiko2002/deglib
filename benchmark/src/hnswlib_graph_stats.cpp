#include <omp.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <tsl/robin_set.h>

#include <algorithm>

#include "hnswlib.h"
#include "hnsw/utils.h"
#include "stopwatch.h"



static uint32_t compute_reachablity_count(hnswlib::HierarchicalNSW<float>* graph) {

    auto stopw = StopW();
    const auto graph_size = (int64_t) graph->cur_element_count;
    uint32_t reachable_count = 0;
    uint32_t tested_count = 0;

    #pragma omp parallel for
    for (int64_t id = 0; id < graph_size; id++) {
        auto target_id = (tableint) id;
        auto target_data = graph->getDataByInternalId(target_id);
        auto found = false;

        // try a simple search first to find the target vertex
        graph->setEf(100);
        {
            auto result_queue = graph->searchKnn(target_data, 100);
            while (result_queue.empty() == false)
            {
                if (result_queue.top().second == target_id) {
                    found = true;
                    break;
                }
                result_queue.pop();
            }
        }

        // if the simple search was not successful use flood fill to find the target vertex
        if(found == false) {

            // search the node in the higher layer to find the best entrance position on the base layer
            auto currObj = graph->enterpoint_node_;
            auto curdist = graph->fstdistfunc_(target_data, graph->getDataByInternalId(graph->enterpoint_node_), graph->dist_func_param_);
            for (int level = graph->maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int* data = (unsigned int*)graph->get_linklist(currObj, level);
                    int size = graph->getListCount(data);
                    
                    tableint* datal = (tableint*)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > graph->max_elements_) throw std::runtime_error("cand error");
                        float d = graph->fstdistfunc_(target_data, graph->getDataByInternalId(cand), graph->dist_func_param_);

                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            // flood fill from this entrance position and try to reach the target_id
            auto checked_ids = std::vector<bool>(graph_size);
            auto check = std::vector<tableint>();
            auto check_next = std::vector<tableint>();

            // start with the first node
            checked_ids[currObj] = true;
            check.emplace_back(currObj);
            
            // repeat as long as we have nodes to check
            found = currObj == target_id;
            auto check_ptr = &check;
            auto check_next_ptr = &check_next;
            while(check_ptr->size() > 0 && found == false) {	

                // neighbors which will be checked next round
                check_next_ptr->clear();

                // get the neighbors to check next
                for (size_t c = 0; c < check_ptr->size() && found == false; c++)
                {
                    auto check_index = check_ptr->at(c);                
                    unsigned int* data = (unsigned int*)graph->get_linklist0(check_index);
                    int size = graph->getListCount(data);
                        
                    tableint* datal = (tableint*)(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > graph->max_elements_) throw std::runtime_error("cand error");

                        if(cand == target_id)
                            found = true;

                        if(checked_ids[cand] == false) {
                            checked_ids[cand] = true;
                            check_next_ptr->emplace_back(cand);
                        }
                    }
                }

                auto buffer = check_ptr;
                check_ptr = check_next_ptr;
                check_next_ptr = buffer;
            }
        }

        #pragma omp critical
        {
            if(found)
                reachable_count++;

            tested_count++;
            if(tested_count % 10000 == 0)
                fmt::print("Seed reachability is {:7d} after checking {:7d} of {:7d} vertices after {:4d}s\n", reachable_count, tested_count, graph_size, stopw.getElapsedTimeMicro() / 1000000);
        }
    }
    
    fmt::print("Seed Reachablity is {} out of {}\n", reachable_count, graph_size);
    return reachable_count;
}

struct VertexReach {
  uint32_t vertex_id;
  uint32_t reach_count;
  std::vector<bool> reachable_ids;
};

static float compute_avg_reach(hnswlib::HierarchicalNSW<float>* graph) {
    const auto graph_size = (tableint) graph->cur_element_count;
    auto stopw = StopW();

    // remember those vertices which have a very high reach
    uint32_t best_vertex_reach = 0;                             
    auto vertices_reach = std::vector<VertexReach>();    
    auto index_of_vertex_reach = std::vector<uint32_t>(graph_size);    
    std::fill(index_of_vertex_reach.begin(), index_of_vertex_reach.end(), graph_size);

    // find the reach of each vertex
    uint64_t counter = 0;
    uint64_t avg_reach = 0;
    for (tableint entry_id = 0; entry_id < graph_size; entry_id++) {
        
        // flood fill from this entrance position and try to reach the target_id
        auto checked_ids = std::vector<bool>(graph_size);
        auto check = std::vector<tableint>();
        auto check_next = std::vector<tableint>();

        // start with the first node
        checked_ids[entry_id] = true;
        check.emplace_back(entry_id);
        
        // we try to speed up the process by reaching a vertex which can reach a lot of other vertices
        uint32_t best_reach_vertex_index = 0;
        uint32_t best_reach_vertex_reach = 0;

        // repeat as long as we have nodes to check
        auto check_ptr = &check;
        auto check_next_ptr = &check_next;
		while(check_ptr->size() > 0 && best_reach_vertex_reach < graph_size) {	

            // neighbors which will be checked next round
            check_next_ptr->clear();

            // get the neighbors to check next
            for (size_t c = 0; c < check_ptr->size() && best_reach_vertex_reach < graph_size; c++) {
                const auto check_index = check_ptr->at(c);
                unsigned int* data = (unsigned int*)graph->get_linklist0(check_index);
                const int size = graph->getListCount(data);

                if(size == 0)
                    fmt::print("zero out-degree for vertex {}\n", check_index);

                const tableint* neighbor_indices = (tableint*)(data + 1);
                for (int n = 0; n < size; n++) {
                    const tableint neighbor_index = neighbor_indices[n];
                    if (neighbor_index < 0 || neighbor_index > graph_size) 
                        throw std::runtime_error("cand error");

                    // consider only neighbors which have not been checked yet
                    if(checked_ids[neighbor_index] == false) {
                        checked_ids[neighbor_index] = true;
                        check_next_ptr->emplace_back(neighbor_index);

                        // is the neighbor connected to a vertex which can reach a lot of other vertices
                        const auto vertex_reach_index = index_of_vertex_reach[neighbor_index];
                        if(vertex_reach_index < graph_size) {

                            // found a vertex which can reach all other vertices
                            const auto& neighbor_reach = vertices_reach[vertex_reach_index];
                            if(neighbor_reach.reach_count == graph_size) {
                                best_reach_vertex_index = vertex_reach_index;
                                best_reach_vertex_reach = graph_size;
                                break;
                            }

                            // found one of the best vertices or a vertex which can reach the best
                            if(neighbor_reach.reach_count > best_reach_vertex_reach) {
                                best_reach_vertex_reach = neighbor_reach.reach_count;
                                best_reach_vertex_index = vertex_reach_index;

                                // copy the reach of the best
                                const auto& best_vertex_checked_ids = neighbor_reach.reachable_ids;
                                for (size_t b = 0; b < graph_size; b++) 
                                    checked_ids[b] = checked_ids[b] | best_vertex_checked_ids[b];
                            }
                        }
                    }
                }
            }

            auto buffer = check_ptr;
            check_ptr = check_next_ptr;
            check_next_ptr = buffer;
        }

        // found path to a vertex which can reach every other vertex
        if(best_reach_vertex_reach == graph_size) {
            index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            avg_reach += graph_size;

        } else {

            // how many nodes have been checked
            uint32_t reach_count =  0;
            for (size_t i = 0; i < graph_size; i++)
                reach_count += checked_ids[i];
            avg_reach += reach_count;
            
            // is this a new best vertex?
            if(best_vertex_reach < reach_count) {
                best_vertex_reach = reach_count;
                index_of_vertex_reach[entry_id] = (uint32_t) vertices_reach.size();
                vertices_reach.emplace_back(entry_id, reach_count, std::move(checked_ids));
                // fmt::print("Current best vertex {} with reach of {}\n", entry_id, reach_count);
            } else if(best_reach_vertex_reach > 0) {
                index_of_vertex_reach[entry_id] = best_reach_vertex_index;
            } else {
                index_of_vertex_reach[entry_id] = (uint32_t) vertices_reach.size();
                vertices_reach.emplace_back(entry_id, reach_count, std::move(checked_ids));
            }
        }


        counter++;
        if(counter % 10000 == 0)
            fmt::print("Avg reach is {:.2f} after checking {:7d} of {:7d} vertices after {:4d}s\n", ((float)avg_reach)/counter, counter, graph_size, stopw.getElapsedTimeMicro() / 1000000);
    }  
    return ((float)avg_reach)/graph_size;
}

static void compute_stats(const char* graph_file, const char* top_list_file, const int feature_dims) {
    fmt::print("Compute graph stats of {}\n", graph_file);

    auto l2space = hnswlib::L2Space(feature_dims);
    auto graph = new hnswlib::HierarchicalNSW<float>(&l2space, graph_file, false);
    auto graph_size = graph->cur_element_count;
    auto max_edges_per_node = graph->maxM0_;
    
    // compute the graph quality
    float perfect_neighbor_ratio = 0;
    float avg_edge_count = 0;
    {
        
        size_t top_list_dims;
        size_t top_list_count;
        const auto all_top_list = ivecs_read(top_list_file, top_list_dims, top_list_count);
        fmt::print("Load TopList from file {} with {} elements and k={}\n", top_list_file, top_list_count, top_list_dims);

        if(top_list_count != graph_size) {
            fmt::print(stderr, "The number of elements in the TopList file is different than in the graph: {} vs {}\n", top_list_count, graph_size);
            return;
        }

        if(top_list_dims < max_edges_per_node) {
            fmt::print(stderr, "Edges per verftex {} is higher than the TopList size = {} \n", max_edges_per_node, top_list_dims);
            return;
        }

        uint64_t perfect_neighbor_count = 0;
        uint64_t total_neighbor_count = 0;
        for (uint32_t n = 0; n < graph_size; n++) {
            auto linklist_data = graph->get_linklist_at_level(n, 0);
            auto edges_per_node = graph->getListCount(linklist_data);
            auto neighbor_indices = (tableint*)(linklist_data + 1);

            // get top list of this node
            auto top_list = all_top_list + n * top_list_dims;
            if(top_list_dims < edges_per_node) {
                fmt::print("TopList for {} is not long enough. has {} elements expected {}\n", n, top_list_dims, edges_per_node);
                edges_per_node = (uint16_t) top_list_dims;
            }
            total_neighbor_count += edges_per_node;

            // check if every neighbor is from the perfect neighborhood
            for (uint32_t e = 0; e < edges_per_node; e++) {
                auto neighbor_index = neighbor_indices[e];

                // find in the neighbor in the first few elements of the top list
                for (uint32_t i = 0; i < edges_per_node; i++) {
                    if(neighbor_index == top_list[i]) {
                        perfect_neighbor_count++;
                        break;
                    }
                }
            }
        }

        perfect_neighbor_ratio = ((float) perfect_neighbor_count) / total_neighbor_count;
        avg_edge_count = ((float) total_neighbor_count) / graph_size;

        delete all_top_list;
    }

    // compute the min and max out degree
    uint16_t min_out = (uint16_t) max_edges_per_node;
    uint16_t max_out = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto linklist_data = graph->get_linklist_at_level(n, 0);
        auto edges_per_node = graph->getListCount(linklist_data);

        if(edges_per_node < min_out)
            min_out = edges_per_node;
        if(max_out < edges_per_node)
            max_out = edges_per_node;
    }

    // compute the in_degree per vertex
    auto in_degree_count = std::vector<uint32_t>(graph_size);
    for (uint32_t n = 0; n < graph_size; n++) {
        auto linklist_data = graph->get_linklist_at_level(n, 0);
        auto edges_per_node = graph->getListCount(linklist_data);
        auto neighbor_indices = (tableint*)(linklist_data + 1);

        for (uint32_t e = 0; e < edges_per_node; e++) {
            auto neighbor_index = neighbor_indices[e];
            in_degree_count[neighbor_index]++;
        }
    }

    // compute the min and max in degree
    uint32_t min_in = in_degree_count[0];
    uint32_t max_in = 0;
    uint32_t source_nodes = 0;
    for (uint32_t n = 0; n < graph_size; n++) {
        auto in_degree = in_degree_count[n];

        if(in_degree < min_in)
            min_in = in_degree;
        if(max_in < in_degree)
            max_in = in_degree;
        if(in_degree == 0) 
            source_nodes++;
    }

    const auto max_level = graph->maxlevel_;
    const auto& level_of_elements = graph->element_levels_;
    auto vertex_count_per_level = std::vector<uint32_t>(max_level+1);
    for (size_t i = 0; i < level_of_elements.size(); i++) 
        vertex_count_per_level[level_of_elements[i]]++;
    for (int i = max_level; i >= 1; i--) // vertices on the higher level exist on the level below
        vertex_count_per_level[i-1] += vertex_count_per_level[i];
    fmt::print("GQ {}, avg degree {}, min_out {}, max_out {}, min_in {}, max_in {}, source vertices {}, vertices per layer ({})\n", perfect_neighbor_ratio, avg_edge_count, min_out, max_out, min_in, max_in, source_nodes, fmt::join(vertex_count_per_level, ","));

    auto reachability_count = compute_reachablity_count(graph);
    auto avg_reach = compute_avg_reach(graph);
    fmt::print("search reachability count {}, exploration avg reach {:.2f}\n",  reachability_count, avg_reach);
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

    const auto data_path = std::filesystem::path(DATA_PATH);

    // sift1m_ef_500_M_24 -> GQ 0.36061648, avg degree 29.743711, min_out 1, max_out 48, min_in 0, max_in 182, source nodes 3, search reachability count 999997, exploration avg reach 999997, node count 1000000, nodes per layer (999998,41583,1763,68)
    // ef_800_M_40_maxM0_50 -> GQ 0.34640437, avg degree 32.855633, min_out 1, max_out 50, min_in 1, max_in 151, source vertices 0, search reachability count 1000000, exploration avg reach 1000000.00, vertices per layer (1000000,24834,568,18,1)
    // const auto graph_file = (data_path / "hnsw" / "ef_800_M_40_maxM0_50.hnsw").string(); 
    // const auto top_list_file = (data_path / "SIFT1M" / "sift_base_top1000.ivecs").string();
    // const int feature_dims = 128;

    // glove-100_ef_2500_M_25 -> GQ 0.23849423, avg degree 18.42118, min_out 1, max_out 50, min_in 0, max_in 1866, source nodes 30686, search reachability count 1152468, exploration avg reach 1152424, node count 1183514, nodes per layer (1183510,47260,1896,64)
    // glove-100_ef_700_M_50_maxM0_60 -> GQ 0.28946307, avg degree 14.822003, min_out 1, max_out 60, min_in 0, max_in 2161, source vertices 41294, vertices per layer (1183514,23565,433,11)
    const auto graph_file = (data_path / "hnsw" / "glove-100_ef_700_M_50_maxM0_60.hnsw").string(); 
    const auto top_list_file  = (data_path / "glove-100" / "glove-100_base_top1000.ivecs").string(); 
    const int feature_dims = 100;

    // const auto graph_file = (data_path / "2dgraph_ef_500_M_4.hnsw").string(); 
    // const auto top_list_file  = (data_path / "base_top13.ivecs").string(); 
    // const int feature_dims = 2;

    // const auto graph_file = (data_path / "hnsw" / "uqv_ef_200_M_10_maxM0_40.hnsw").string(); 
    // const auto top_list_file  = (data_path / "uqv" / "uqv_base_top1000.ivecs").string(); 
    // const int feature_dims = 256;

    // ef_900_M_50_maxM0_80 -> GQ 0.33040112, avg degree 21.918083, min_out 1, max_out 80, min_in 0, max_in 243, source vertices 34, search reachability count 94951, exploration avg reach 94951.00, vertices per layer (94987,1848,36)
    // const auto graph_file = (data_path / "hnsw" / "ef_900_M_50_maxM0_80.hnsw").string(); 
    // const auto top_list_file  = (data_path / "enron" / "enron_base_top1000.ivecs").string(); 
    // const int feature_dims = 1368;

    compute_stats(graph_file.c_str(), top_list_file.c_str(), feature_dims);

    fmt::print("Test OK\n");
    return 0;
}