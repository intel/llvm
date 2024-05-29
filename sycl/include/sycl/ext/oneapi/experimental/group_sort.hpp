//==--------- group_sort.hpp --- SYCL extension group sorting algorithm-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)

#include "group_helpers_sorters.hpp" // for default_sorter, group_with_sc...

#include <sycl/detail/pi.h>            // for PI_ERROR_INVALID_DEVICE
#include <sycl/detail/type_traits.hpp> // for is_generic_group
#include <sycl/exception.hpp>          // for sycl_category, exception

#include <stddef.h>     // for size_t
#include <system_error> // for error_code
#include <type_traits>  // for enable_if_t, decay_t, false_type
#include <utility>      // for declval

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

// ---- traits
template <typename T, typename = void> struct has_difference_type {};

template <typename T>
struct has_difference_type<T, std::void_t<typename T::difference_type>>
    : std::true_type {};

template <typename T> struct has_difference_type<T *> : std::true_type {};

template <typename Sorter, typename Group, typename Val, typename = void>
struct is_sorter_impl {
  template <typename G>
  using is_expected_return_type =
      typename std::is_same<Val, decltype(std::declval<Sorter>()(
                                     std::declval<G>(), std::declval<Val>()))>;

  template <typename G = Group>
  static decltype(std::integral_constant<bool,
                                         is_expected_return_type<G>::value &&
                                             sycl::is_group_v<G>>{})
  test(int);

  template <typename = Group> static std::false_type test(...);
};

template <typename Sorter, typename Group,
          typename Ptr> // multi_ptr has difference_type and don't have other
                        // iterator's fields
struct is_sorter_impl<Sorter, Group, Ptr,
                      std::void_t<typename has_difference_type<Ptr>::type>> {
  template <typename G = Group>
  static decltype(std::declval<Sorter>()(std::declval<G>(), std::declval<Ptr>(),
                                         std::declval<Ptr>()),
                  sycl::detail::is_generic_group<G>{})
  test(int);

  template <typename = Group> static std::false_type test(...);
};

template <typename Sorter, typename Group, typename ValOrPtr>
struct is_sorter : decltype(is_sorter_impl<Sorter, Group, ValOrPtr>::test(0)) {
};

template <typename Sorter, typename Group, typename Key, typename Value>
struct is_key_value_sorter_impl {
  template <typename G>
  using is_expected_return_type =
      typename std::is_same<std::tuple<Key, Value>,
                            decltype(std::declval<Sorter>()(
                                std::declval<G>(), std::declval<Key>(),
                                std::declval<Value>()))>;

  template <typename G = Group>
  static decltype(std::integral_constant<bool,
                                         is_expected_return_type<G>::value &&
                                             sycl::is_group_v<G>>{})
  test(int);

  template <typename = Group> static std::false_type test(...);
};

template <typename Sorter, typename Group, typename Key, typename Value>
struct is_key_value_sorter
    : decltype(is_key_value_sorter_impl<Sorter, Group, Key, Value>::test(0)){};

template <typename Property> struct is_data_placement_property {
  static constexpr bool value = std::is_same_v<Property, detail::is_blocked> ||
                                std::is_same_v<Property, detail::is_striped>;
};

} // namespace detail

// ---- sort_over_group
template <typename Group, typename T, typename Sorter>
std::enable_if_t<detail::is_sorter<Sorter, Group, T>::value, T>
sort_over_group(Group group, T value, Sorter sorter) {
#ifdef __SYCL_DEVICE_ONLY__
  return sorter(group, value);
#else
  (void)group;
  (void)value;
  (void)sorter;
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group algorithms are not supported on host device.");
#endif
}

template <typename Group, typename T, typename Compare, size_t Extent>
std::enable_if_t<!detail::is_sorter<Compare, Group, T>::value, T>
sort_over_group(experimental::group_with_scratchpad<Group, Extent> exec,
                T value, Compare comp) {
  return sort_over_group(
      exec.get_group(), value,
      default_sorters::group_sorter<T, 1, Compare>(exec.get_memory(), comp));
}

template <typename Group, typename T, size_t Extent>
std::enable_if_t<sycl::is_group_v<std::decay_t<Group>>, T>
sort_over_group(experimental::group_with_scratchpad<Group, Extent> exec,
                T value) {
  return sort_over_group(exec.get_group(), value,
                         default_sorters::group_sorter<T>(exec.get_memory()));
}

template <typename Group, typename T, std::size_t ElementsPerWorkItem,
          typename Sorter, typename Properties = detail::is_blocked>
std::enable_if_t<detail::is_data_placement_property<Properties>::value, void>
sort_over_group(Group g, sycl::span<T, ElementsPerWorkItem> values,
                Sorter sorter, Properties properties = {}) {
#ifdef __SYCL_DEVICE_ONLY__
  return sorter(g, values, properties);
#else
  (void)g;
  (void)values;
  (void)sorter;
  (void)properties;
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group algorithms are not supported on host device.");
#endif
}

template <typename Group, typename T, std::size_t Extent,
          std::size_t ElementsPerWorkItem,
          typename Properties = detail::is_blocked>
std::enable_if_t<detail::is_data_placement_property<Properties>::value, void>
sort_over_group(experimental::group_with_scratchpad<Group, Extent> exec,
                sycl::span<T, ElementsPerWorkItem> values,
                Properties properties = {}) {
  return sort_over_group(
      exec.get_group(), values,
      default_sorters::group_sorter<T, ElementsPerWorkItem>(exec.get_memory()),
      properties);
}

template <typename Group, typename T, std::size_t Extent,
          std::size_t ElementsPerWorkItem, typename Compare,
          typename Properties = detail::is_blocked>
std::enable_if_t<!detail::is_data_placement_property<Compare>::value &&
                     detail::is_data_placement_property<Properties>::value,
                 void>
sort_over_group(experimental::group_with_scratchpad<Group, Extent> exec,
                sycl::span<T, ElementsPerWorkItem> values, Compare comp,
                Properties properties = {}) {
  return sort_over_group(
      exec.get_group(), values,
      default_sorters::group_sorter<T, ElementsPerWorkItem, Compare>(
          exec.get_memory(), comp),
      properties);
}

// ---- joint_sort
template <typename Group, typename Iter, typename Sorter>
std::enable_if_t<detail::is_sorter<Sorter, Group, Iter>::value, void>
joint_sort(Group group, Iter first, Iter last, Sorter sorter) {
#ifdef __SYCL_DEVICE_ONLY__
  sorter(group, first, last);
#else
  (void)group;
  (void)first;
  (void)last;
  (void)sorter;
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group algorithms are not supported on host device.");
#endif
}

template <typename Group, typename Iter, typename Compare, size_t Extent>
std::enable_if_t<!detail::is_sorter<Compare, Group, Iter>::value, void>
joint_sort(experimental::group_with_scratchpad<Group, Extent> exec, Iter first,
           Iter last, Compare comp) {
  joint_sort(exec.get_group(), first, last,
             default_sorters::joint_sorter<Compare>(exec.get_memory(), comp));
}

template <typename Group, typename Iter, size_t Extent>
std::enable_if_t<sycl::is_group_v<std::decay_t<Group>>, void>
joint_sort(experimental::group_with_scratchpad<Group, Extent> exec, Iter first,
           Iter last) {
  joint_sort(exec.get_group(), first, last,
             default_sorters::joint_sorter<>(exec.get_memory()));
}

template <typename Group, typename T, typename U, typename Sorter>
std::enable_if_t<detail::is_key_value_sorter<Sorter, Group, T, U>::value,
                 std::tuple<T, U>>
sort_key_value_over_group(Group g, T key, U value, Sorter sorter) {
#ifdef __SYCL_DEVICE_ONLY__
  return sorter(g, key, value);
#else
  (void)g;
  (void)key;
  (void)value;
  (void)sorter;
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group algorithms are not supported on host device.");
#endif
}

template <typename Group, typename T, typename U, typename Compare,
          std::size_t Extent>
std::enable_if_t<!detail::is_key_value_sorter<Compare, Group, T, U>::value,
                 std::tuple<T, U>>
sort_key_value_over_group(
    experimental::group_with_scratchpad<Group, Extent> exec, T key, U value,
    Compare comp) {
  return sort_key_value_over_group(
      exec.get_group(), key, value,
      default_sorters::group_key_value_sorter<T, U, Compare>(exec.get_memory(),
                                                             comp));
}

template <typename T, typename U, typename Group, std::size_t Extent>
std::enable_if_t<sycl::is_group_v<std::decay_t<Group>>, std::tuple<T, U>>
sort_key_value_over_group(
    experimental::group_with_scratchpad<Group, Extent> exec, T key, U value) {
  return sort_key_value_over_group(
      exec.get_group(), key, value,
      default_sorters::group_key_value_sorter<T, U>(exec.get_memory()));
}

// key value sorting
template <std::size_t ElementsPerWorkItem, typename Group, typename T,
          typename U, typename ArraySorter,
          typename Property = detail::is_blocked>
std::enable_if_t<detail::is_data_placement_property<Property>::value, void>
sort_key_value_over_group(Group group, sycl::span<T, ElementsPerWorkItem> keys,
                          sycl::span<U, ElementsPerWorkItem> values,
                          ArraySorter array_sorter, Property property = {}) {
  array_sorter(group, keys, values, property);
}

template <typename Group, typename T, typename U, std::size_t Extent,
          std::size_t ElementsPerWorkItem, typename Compare,
          typename Property = detail::is_blocked>
std::enable_if_t<!detail::is_data_placement_property<Compare>::value &&
                     detail::is_data_placement_property<Property>::value,
                 void>
sort_key_value_over_group(
    experimental::group_with_scratchpad<Group, Extent> exec,
    sycl::span<T, ElementsPerWorkItem> keys,
    sycl::span<U, ElementsPerWorkItem> values, Compare comp,
    Property property = {}) {
  return experimental::sort_key_value_over_group(
      exec.get_group(), keys, values,
      typename experimental::default_sorters::group_key_value_sorter<
          T, U, Compare, ElementsPerWorkItem>(exec.get_memory(), comp),
      property);
}

// TODO: Check for property type
template <typename Group, typename T, typename U, std::size_t Extent,
          std::size_t ElementsPerWorkItem,
          typename Property = detail::is_blocked>
std::enable_if_t<detail::is_data_placement_property<Property>::value, void>
sort_key_value_over_group(
    experimental::group_with_scratchpad<Group, Extent> exec,
    sycl::span<T, ElementsPerWorkItem> keys,
    sycl::span<U, ElementsPerWorkItem> values, Property property = {}) {
  return experimental::sort_key_value_over_group(
      exec.get_group(), keys, values,
      typename experimental::default_sorters::group_key_value_sorter<
          T, U, std::less<>, ElementsPerWorkItem>(exec.get_memory()),
      property);
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
#endif
