#pragma once

#include "search.h"

namespace deglib::graph
{

class MutableGraph : public deglib::search::SearchGraph
{
  public:    

   /**
    * Add a new node. The neighbor indizies will be prefilled with a self-loop, the weights will be 0.
    * 
    * @return the internal index of the new node
    */
    virtual uint32_t addNode(const uint32_t external_label, const std::byte* feature_vector) = 0;

   /**
    * Swap a neighbor with another neighbor and its weight.
    * 
    * @param internal_index node index which neighbors should be changed
    * @param from_neighbor_index neighbor index to remove
    * @param to_neighbor_index neighbor index to add
    * @param to_neighbor_weight weight of the neighbor to add
    * @return true if the from_neighbor_index was found and changed
    */
    virtual bool changeEdge(const uint32_t internal_index, const uint32_t from_neighbor_index, const uint32_t to_neighbor_index, const float to_neighbor_weight) = 0;


    /**
     * Change all edges of a node.
     * The neighbor indizies/weights and feature vectors will be copied.
     * The neighbor array need to have enough neighbors to match the edge-per-node count of the graph.
     * The indizies in the neighbor_indizies array must be sorted.
     */
    virtual void changeEdges(const uint32_t internal_index, const uint32_t* neighbor_indizies, const float* neighbor_weights) = 0;


    /**
     * 
     */
    virtual const float* getNeighborWeights(const uint32_t internal_index) const = 0;    

};

}  // end namespace deglib::graph
