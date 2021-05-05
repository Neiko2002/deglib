/*****************************************************
 * Iterating over the elements of a pointer
 * Reference
 *https://stackoverflow.com/questions/25060051/how-to-perform-a-range-based-c11-for-loop-on-char-argv/25060389#25060389
 *****************************************************/

template <typename T>
struct array_view {
  T *first, *last;
  array_view(T* a, std::size_t n) : first(a), last(a + n) {}
  T* begin() const { return first; }
  T* end() const { return last; }
};

template <typename T>
array_view<T> view_array(T* a, std::size_t n) {
  return array_view<T>(a, n);
}