//==------------ unordered_multimap.hpp ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//
#pragma once
#include <detail/hashers.hpp>
#include <emhash/hash_table8.hpp>

#include <utility>
#include <vector>
namespace sycl {
inline namespace _V1 {
namespace detail {
// A simple implementation of an unordered multimap using a fast hashmap with
// the value type being a vector.
template <typename Key, typename Value> class UnorderedMultimap {
private:
  emhash8::HashMap<Key, std::vector<Value>> map;

  template <typename MapIterator, typename VectorIterator>
  class multimap_iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<const Key, Value>;
    using reference = value_type;
    using pointer = void;
    using difference_type = std::ptrdiff_t;

  private:
    MapIterator map_it, map_end;
    VectorIterator vector_it;

    void skip_empty_vectors() {
      while (map_it != map_end && vector_it == map_it->second.end()) {
        ++map_it;
        if (map_it != map_end)
          vector_it = map_it->second.begin();
      }
    }

  public:
    multimap_iterator(MapIterator m_it, MapIterator m_end)
        : map_it(m_it), map_end(m_end) {
      if (map_it != map_end) {
        vector_it = map_it->second.begin();
        skip_empty_vectors();
      }
    }

    multimap_iterator(MapIterator m_it, MapIterator m_end, VectorIterator v_it)
        : map_it(m_it), map_end(m_end), vector_it(v_it) {}

    reference operator*() const { return {map_it->first, *vector_it}; }

    multimap_iterator &operator++() {
      ++vector_it;
      skip_empty_vectors();
      return *this;
    }

    bool operator==(const multimap_iterator &other) const {
      return map_it == other.map_it &&
             (map_it == map_end || vector_it == other.vector_it);
    }
    bool operator!=(const multimap_iterator &other) const {
      return !(*this == other);
    }

    MapIterator get_map_iterator() const { return map_it; }
    VectorIterator get_vector_iterator() const { return vector_it; }
  };

public:
  using iterator = multimap_iterator<
      typename emhash8::HashMap<Key, std::vector<Value>>::iterator,
      typename std::vector<Value>::iterator>;
  using const_iterator = multimap_iterator<
      typename emhash8::HashMap<Key, std::vector<Value>>::const_iterator,
      typename std::vector<Value>::const_iterator>;

  void insert(const Key &key, const Value &value) { map[key].push_back(value); }

  size_t count(const Key &key) const {
    auto it = map.find(key);
    return (it != map.end()) ? it->second.size() : 0;
  }

  template <typename... Args> void emplace(const Key &key, Args &&...args) {
    map[key].emplace_back(std::forward<Args>(args)...);
  }

  iterator find(const Key &key) {
    auto map_it = map.find(key);
    if (map_it != map.end() && !map_it->second.empty())
      return iterator(map_it, map.end(), map_it->second.begin());
    return end();
  }

  const_iterator find(const Key &key) const {
    auto map_it = map.find(key);
    if (map_it != map.end() && !map_it->second.empty())
      return const_iterator(map_it, map.end(), map_it->second.begin());
    return end();
  }

  void erase(const Key &key) { map.erase(key); }

  iterator erase(iterator it) {
    auto map_it = it.get_map_iterator();
    auto vector_it = it.get_vector_iterator();
    vector_it = map_it->second.erase(vector_it);
    if (map_it->second.empty()) {
      map_it = map.erase(map_it);
      return iterator(map_it, map.end());
    }
    return iterator(map_it, map.end(), vector_it);
  }

  std::pair<iterator, iterator> equal_range(const Key &key) {
    auto map_it = map.find(key);
    if (map_it == map.end())
      return {end(), end()};
    return {iterator(map_it, map.end(), map_it->second.begin()),
            iterator(map_it, map.end(), map_it->second.end())};
  }

  std::vector<Value> &operator[](const Key &key) { return map[key]; }

  iterator begin() { return iterator(map.begin(), map.end()); }
  iterator end() { return iterator(map.end(), map.end()); }
  const_iterator begin() const {
    return const_iterator(map.begin(), map.end());
  }
  const_iterator end() const { return const_iterator(map.end(), map.end()); }
};

} // namespace detail
} // namespace _V1
} // namespace sycl
