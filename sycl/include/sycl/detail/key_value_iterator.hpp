//==------------ key_value_iterator.hpp ------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file includes key/value iterator implementation used for group_sort
// algorithms.
//

#pragma once
#include <iterator>
#include <tuple>
#include <utility>

namespace sycl {
inline namespace _V1 {
namespace detail {

template <typename T1, typename T2> class key_value_iterator {
public:
  key_value_iterator(T1 *Keys, T2 *Values) : KeyValue{Keys, Values} {}

  using difference_type = std::ptrdiff_t;
  using value_type = std::tuple<T1, T2>;
  using reference = std::tuple<T1 &, T2 &>;
  using pointer = std::tuple<T1 *, T2 *>;
  using iterator_category = std::random_access_iterator_tag;

  reference operator*() const {
    return std::tie(*(std::get<0>(KeyValue)), *(std::get<1>(KeyValue)));
  }

  reference operator[](difference_type i) const { return *(*this + i); }

  difference_type operator-(const key_value_iterator &it) const {
    return std::get<0>(KeyValue) - std::get<0>(it.KeyValue);
  }

  key_value_iterator &operator+=(difference_type i) {
    KeyValue =
        std::make_tuple(std::get<0>(KeyValue) + i, std::get<1>(KeyValue) + i);
    return *this;
  }
  key_value_iterator &operator-=(difference_type i) { return *this += -i; }
  key_value_iterator &operator++() { return *this += 1; }
  key_value_iterator &operator--() { return *this -= 1; }
  std::tuple<T1 *, T2 *> base() const { return KeyValue; }
  key_value_iterator operator++(int) {
    key_value_iterator it(*this);
    ++(*this);
    return it;
  }
  key_value_iterator operator--(int) {
    key_value_iterator it(*this);
    --(*this);
    return it;
  }

  key_value_iterator operator-(difference_type i) const {
    key_value_iterator it(*this);
    return it -= i;
  }
  key_value_iterator operator+(difference_type i) const {
    key_value_iterator it(*this);
    return it += i;
  }
  friend key_value_iterator operator+(difference_type i,
                                      const key_value_iterator &it) {
    return it + i;
  }

  bool operator==(const key_value_iterator &it) const {
    return *this - it == 0;
  }

  bool operator!=(const key_value_iterator &it) const { return !(*this == it); }
  bool operator<(const key_value_iterator &it) const { return *this - it < 0; }
  bool operator>(const key_value_iterator &it) const { return it < *this; }
  bool operator<=(const key_value_iterator &it) const { return !(*this > it); }
  bool operator>=(const key_value_iterator &it) const { return !(*this < it); }

private:
  std::tuple<T1 *, T2 *> KeyValue;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
