#pragma once

#include <queue>

namespace deglib::search
{

#pragma pack(2)
class ObjectDistance
{
    uint32_t id_;
    float distance_;

  public:
    ObjectDistance(const uint32_t id, const float distance) : id_(id), distance_(distance) {}

    inline const uint32_t getId() const { return id_; }

    inline const float getDistance() const { return distance_; }

    inline bool operator==(const ObjectDistance& o) const { return (distance_ == o.distance_) && (id_ == o.id_); }

    inline bool operator<(const ObjectDistance& o) const
    {
        if (distance_ == o.distance_)
        {
            return id_ < o.id_;
        }
        else
        {
            return distance_ < o.distance_;
        }
    }

    inline bool operator>(const ObjectDistance& o) const
    {
        if (distance_ == o.distance_)
        {
            return id_ > o.id_;
        }
        else
        {
            return distance_ > o.distance_;
        }
    }
};
#pragma pack()

/**
 * priority queue with access to the internal data.
 * therefore access to the unsorted data is possible.
 * 
 * https://stackoverflow.com/questions/4484767/how-to-iterate-over-a-priority-queue
 * https://www.linuxtopia.org/online_books/programming_books/c++_practical_programming/c++_practical_programming_189.html
 */
template<class Compare>
class PQV : public std::vector<ObjectDistance> {
  Compare comp;
  public:
    PQV(Compare cmp = Compare()) : comp(cmp) {
      std::make_heap(this->begin(),this->end(), comp);
    }

    const ObjectDistance& top() { return this->front(); }

    template <class... _Valty>
    void emplace(_Valty&&... _Val) {
      this->emplace_back(std::forward<_Valty>(_Val)...);
      std::push_heap(this->begin(), this->end(), comp);
    }

    void push(const ObjectDistance& x) {
      this->push_back(x);
      std::push_heap(this->begin(),this->end(), comp);
    }

    void pop() {
      std::pop_heap(this->begin(),this->end(), comp);
      this->pop_back();
    }
};

typedef PQV<std::less<ObjectDistance>> ResultSet;

// search result set containing node ids and distances
//typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::less<ObjectDistance>> ResultSet;

// set of unchecked node ids
typedef std::priority_queue<ObjectDistance, std::vector<ObjectDistance>, std::greater<ObjectDistance>> UncheckedSet;





class SearchGraph
{
  public:    
    virtual deglib::search::ResultSet yahooSearch(const std::vector<uint32_t>& entry_node_indizies, const float* query, const float eps, const int k)  const = 0;
    virtual const uint32_t getExternalLabel(const uint32_t internal_index) const = 0;
    virtual const uint32_t getInternalIndex(const uint32_t external_label) const = 0;
    virtual const uint8_t getEdgesPerNode() const = 0;
    virtual const size_t size() const = 0;
    virtual const deglib::SpaceInterface<float>& getFeatureSpace() const = 0;
};

} // end namespace deglib::search