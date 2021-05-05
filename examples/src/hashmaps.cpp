//#include <absl/container/flat_hash_map.h>
//#include <fmt/core.h>

#include <algorithm>
#include <map>
#include <memory>
#include <memory_resource>
#include <optional>
#include <unordered_map>
#include <iostream>

/*template <class K, class V>
using PmrFlatHashMap =
    absl::flat_hash_map<K, V, absl::Hash<K>, std::equal_to<>,
                        std::pmr::polymorphic_allocator<std::pair<const K, V>>>;
*/

template <class K, class V>
std::optional<std::reference_wrapper<const V>> linear_find(
    const std::vector<std::pair<K, V>>& v, const K& key) {
  auto it =
      std::ranges::find_if(v, [&](auto&& pair) { return pair.first == key; });
  if (it != v.end()) {
    return it->second;
  }
  return std::nullopt;
}

struct Knot {
  int f;
  std::optional<std::pmr::unordered_map<int, float>> map;
};

int main() {

  // https://stackoverflow.com/questions/49596685/how-do-i-declare-an-array-without-initializing-a-constant/49598580#49598580
  // std::vector<std::byte> buffer{1000};
  // std::array<std::byte, 1001> buffer;
  auto buffer = std::make_unique<std::byte[]>(1000);
  //auto buffer = std::unique_ptr<std::byte[]>(new std::byte[1000]);
  {
    auto resource = std::pmr::monotonic_buffer_resource(
        buffer.get(), 1000, std::pmr::null_memory_resource());
    auto map = std::pmr::unordered_map<int, float>(&resource);
    map.reserve(2);
    // auto map = PmrFlatHashMap<int, float>(&resource);
    map.emplace(1, 2);
    map.emplace(2, 2);
    for (auto&& v : map) {
      std:cout << (void*)std::addressof(v)) << std:endl;
    }
  }
  return sizeof(Knot);
}

auto set_map(Knot& knot) {
  auto map = std::pmr::unordered_map<int, float>();
  map.reserve(2);
  map.emplace(1, 2);
  map.emplace(2, 2);
  return map;
}

auto entry() {
  Knot knot;
  auto& [_, map] = knot;
  map = set_map(knot);
  return sizeof(Knot);
}