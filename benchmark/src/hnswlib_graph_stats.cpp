#include <omp.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <tsl/robin_set.h>

#include <algorithm>

#include "hnswlib.h"
#include "hnsw/utils.h"
#include "stopwatch.h"


static uint32_t compute_reachablity_count(hnswlib::HierarchicalNSW<float>* graph) {

    auto graph_size = graph->cur_element_count;
    uint32_t reachable_count = 0;
    for (tableint target_id = 0; target_id < graph_size; target_id++) {
        auto target_data = graph->getDataByInternalId(target_id);
        tableint currObj = graph->enterpoint_node_;
        float curdist = graph->fstdistfunc_(target_data, graph->getDataByInternalId(graph->enterpoint_node_), graph->dist_func_param_);

        // search the node in the higher layer to find the best entrance position on the base layer
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
        boolean found = currObj == target_id;
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

        if(found)
            reachable_count++;

        if((target_id+1) % 1000 == 0)
            fmt::print("Seed Reachablity is {} out of {} with {} more to check\n", reachable_count, (target_id+1), (graph_size-(target_id+1)));
    }
    
    fmt::print("Seed Reachablity is {} out of {}\n", reachable_count, graph_size);
    return reachable_count;
}

static float compute_avg_reach(hnswlib::HierarchicalNSW<float>* graph, std::vector<bool>& is_source_vertex) {
    auto graph_size = (tableint) graph->cur_element_count;

    // remember those vertices which have a very high reach
    uint32_t best_vertex_checked_count = 0;                             // current highest reach count
    auto best_vertices_checked_ids = std::vector<std::vector<bool>>();  // which vertices can be reach by one of the best vertices
    auto best_reach_indices = std::vector<uint32_t>(graph_size);        // a list of all vertices containing the index to one of the best vertices
    std::fill(best_reach_indices.begin(), best_reach_indices.end(), graph_size);

    // numbers of vertices to be reached starting from each vertex
    auto reach_counts = std::vector<uint32_t>(graph_size);
    uint64_t avg_reach = 0;
    for (tableint entry_id = 0; entry_id < graph_size; entry_id++) {
        
        // flood fill from this entrance position and try to reach the target_id
        auto checked_ids = std::vector<bool>(graph_size);
        auto check = std::vector<tableint>();
        auto check_next = std::vector<tableint>();

        // start with the first node
        checked_ids[entry_id] = true;
        check.emplace_back(entry_id);
        
        // we try to speed up the process by reaching a vertex which can reach all other vertices
        // or we find a vertex which is currently the best vertex and copy its reach
        bool reach_all = false;
        bool reach_best = false;        

        // repeat as long as we have nodes to check
        auto check_ptr = &check;
        auto check_next_ptr = &check_next;
		while(check_ptr->size() > 0 && reach_all == false) {	

            // neighbors which will be checked next round
            check_next_ptr->clear();

            // get the neighbors to check next
            for (size_t c = 0; c < check_ptr->size() && reach_all == false; c++) {
                auto check_index = check_ptr->at(c);
                unsigned int* data = (unsigned int*)graph->get_linklist0(check_index);
                int size = graph->getListCount(data);

                if(size == 0)
                    fmt::print("zero out-degree for vertex {}\n", check_index);

                tableint* datal = (tableint*)(data + 1);
                for (int i = 0; i < size; i++)
                {
                    tableint neighbor_index = datal[i];
                    if (neighbor_index < 0 || neighbor_index > graph_size) throw std::runtime_error("cand error");

                    if(checked_ids[neighbor_index] == false) {
                        checked_ids[neighbor_index] = true;
                        check_next_ptr->emplace_back(neighbor_index);

                        // found a vertex which can reach all other vertices
                        if(reach_counts[neighbor_index] == graph_size) {
                            reach_all = true;
                            break;
                        }

                        // found one of the best vertices or a vertex which can reach the best -> copy the reach of the best
                        if(reach_best == false && reach_all == false) {
                            auto best_reach_index = best_reach_indices[neighbor_index];
                            if(best_reach_index < graph_size) {
                                best_reach_indices[entry_id] = best_reach_index;

                                auto& best_vertex_checked_ids = best_vertices_checked_ids[best_reach_index];
                                for (size_t b = 0; b < graph_size; b++) 
                                    checked_ids[b] = checked_ids[b] | best_vertex_checked_ids[b];
                                reach_best = true;
                            }
                        }
                    }
                }
            }

            auto buffer = check_ptr;
            check_ptr = check_next_ptr;
            check_next_ptr = buffer;
        }

        // how many nodes have been checked
        uint32_t reach_count = reach_all ? graph_size : 0;
        if(reach_all == false)
            for (size_t i = 0; i < graph_size; i++)
                reach_count += checked_ids[i];
        reach_counts[entry_id] = reach_count;
        avg_reach += reach_count;
        
        // is this a new best vertex?
        if(is_source_vertex[entry_id] == false && (reach_count > best_vertex_checked_count || reach_best == false)) {
            best_vertex_checked_count = reach_count;
            best_reach_indices[entry_id] = best_vertices_checked_ids.size();

            if(reach_all)
                std::fill(checked_ids.begin(), checked_ids.end(), true);
            best_vertices_checked_ids.emplace_back(checked_ids);
            fmt::print("Current best vertex {} with reach of {}\n", entry_id, best_vertex_checked_count);
        }

        if((entry_id+1) % 1000 == 0)
            fmt::print("Avg reach is {:.2f} after checking {} of {} vertices\n", ((float)avg_reach)/(entry_id+1), (entry_id+1), graph_size);
    }  
    return ((float)avg_reach)/graph_size;
}

static void compute_stats(const char* graph_file, const char* top_list_file, const int feature_dims) {
    fmt::print("Compute graph stats of {}\n", graph_file);

    auto l2space = hnswlib::L2Space(feature_dims);
    auto graph = new hnswlib::HierarchicalNSW<float>(&l2space, graph_file, false);
    auto graph_size = graph->cur_element_count;
    auto max_edges_per_node = graph->maxM0_;

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
    
    // compute the graph quality
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
    auto perfect_neighbor_ratio = ((float) perfect_neighbor_count) / total_neighbor_count;
    auto avg_edge_count = ((float) total_neighbor_count) / graph_size;

    // compute the min, and max out degree
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

    // compute the min, and max in degree
    auto in_degree_count = std::vector<uint32_t>(graph_size);
    for (uint32_t n = 0; n < graph_size; n++) {
        auto linklist_data = graph->get_linklist_at_level(n, 0);
        auto edges_per_node = graph->getListCount(linklist_data);
        auto neighbor_indices = (tableint*)(linklist_data + 1);

        // fmt::print("Neighbors of vertex {}\n", n);
        for (uint32_t e = 0; e < edges_per_node; e++) {
            // fmt::print("{} = {}\n", e, neighbor_indices[e]);
            auto neighbor_index = neighbor_indices[e];
            in_degree_count[neighbor_index]++;
        }
        // fmt::print("\n");
    }


    uint32_t min_in = in_degree_count[0];
    uint32_t max_in = 0;
    uint32_t source_node_count = 0;
    uint32_t source_nodes = 0;
    auto is_source_vertex = std::vector<bool>(graph_size);
    std::fill(is_source_vertex.begin(), is_source_vertex.end(), false);
    for (uint32_t n = 0; n < graph_size; n++) {
        auto in_degree = in_degree_count[n];

        if(in_degree < min_in)
            min_in = in_degree;
        if(max_in < in_degree)
            max_in = in_degree;

        if(in_degree == 0) {
            is_source_vertex[n] = true;
            source_nodes++;
            // fmt::print("Node {} has zero incoming connections and exists up to layer {} \n", n, graph->element_levels_[n]);
        }
    }

    auto max_level = graph->maxlevel_;
    auto node_per_level_count = std::vector<uint32_t>(max_level+1);
    auto node_levels = graph->element_levels_;
    for (size_t i = 0; i < node_levels.size(); i++) 
        node_per_level_count[node_levels[i]]++;
    for (int i = max_level; i >= 1; i--) // vertices on the higher level exist on the level below
        node_per_level_count[i-1] += node_per_level_count[i];

    // auto reachability_count = 0;//compute_reachablity_count(graph); // Glove 1152424
    auto reachability_count = compute_reachablity_count(graph);
    auto avg_reach = compute_avg_reach(graph, is_source_vertex);

    fmt::print("GQ {}, avg degree {}, min_out {}, max_out {}, min_in {}, max_in {}, source nodes {}, search reachability count {}, exploration avg reach {:.2f}, node count {}, nodes per layer ({})\n", perfect_neighbor_ratio, avg_edge_count, min_out, max_out, min_in, max_in, source_node_count, reachability_count, avg_reach, graph_size, fmt::join(node_per_level_count, ","));
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

    // GQ 0.36061648, avg degree 29.743711, min_out 1, max_out 48, min_in 0, max_in 182, source nodes 3, search reachability count 999997, exploration avg reach 999997, node count 1000000, nodes per layer (999998,41583,1763,68)
    // const auto graph_file = (data_path / "hnsw" / "sift1m_ef_500_M_24.hnsw").string(); 
    // const auto top_list_file = (data_path / "SIFT1M" / "sift_base_top1000.ivecs").string();

    // GQ 0.23849423, avg degree 18.42118, min_out 1, max_out 50, min_in 0, max_in 1866, source nodes 30686, search reachability count 1152468, exploration avg reach 1152424, node count 1183514, nodes per layer (1183510,47260,1896,64)
    // const auto graph_file = (data_path / "hnsw" / "glove-100_ef_2500_M_25.hnsw").string(); 
    // const auto top_list_file  = (data_path / "glove-100" / "glove_base_top1000.ivecs").string(); 

    // const auto graph_file = (data_path / "2dgraph_ef_500_M_4.hnsw").string(); 
    // const auto top_list_file  = (data_path / "base_top13.ivecs").string(); 
    // const int feature_dims = 2;

    const auto graph_file = (data_path / "hnsw" / "uqv_ef_200_M_10_maxM0_40.hnsw").string(); 
    const auto top_list_file  = (data_path / "uqv" / "uqv_base_top1000.ivecs").string(); 
    const int feature_dims = 256;

    

    compute_stats(graph_file.c_str(), top_list_file.c_str(), feature_dims);

    fmt::print("Test OK\n");
    return 0;
}