//==--------- group_sort.hpp --- SYCL extension group sorting algorithm-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/group_sort_impl.hpp>
#include <sycl/detail/type_traits.hpp>
#include <type_traits>

#include "group_helpers_sorters.hpp"

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::experimental {
namespace detail {

// ---- traits
template <typename T, typename = void> struct has_difference_type {};

template <typename T>
struct has_difference_type<T, sycl::detail::void_t<typename T::difference_type>>
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
struct is_sorter_impl<
    Sorter, Group, Ptr,
    sycl::detail::void_t<typename has_difference_type<Ptr>::type>> {
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
} // namespace detail

// ---- sort_over_group
template <typename Group, typename T, typename Sorter>
typename std::enable_if<detail::is_sorter<Sorter, Group, T>::value, T>::type
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
typename std::enable_if<!detail::is_sorter<Compare, Group, T>::value, T>::type
sort_over_group(experimental::group_with_scratchpad<Group, Extent> exec,
                T value, Compare comp) {
  return sort_over_group(
      exec.get_group(), value,
      default_sorters::group_sorter<T, 1, Compare>(exec.get_memory(), comp));
}

template <typename Group, typename T, size_t Extent>
typename std::enable_if<sycl::is_group_v<std::decay_t<Group>>, T>::type
sort_over_group(experimental::group_with_scratchpad<Group, Extent> exec,
                T value) {
  return sort_over_group(exec.get_group(), value,
                         default_sorters::group_sorter<T>(exec.get_memory()));
}

// ---- joint_sort
template <typename Group, typename Iter, typename Sorter>
typename std::enable_if<detail::is_sorter<Sorter, Group, Iter>::value,
                        void>::type
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
typename std::enable_if<!detail::is_sorter<Compare, Group, Iter>::value,
                        void>::type
joint_sort(experimental::group_with_scratchpad<Group, Extent> exec, Iter first,
           Iter last, Compare comp) {
  joint_sort(exec.get_group(), first, last,
             default_sorters::joint_sorter<Compare>(exec.get_memory(), comp));
}

template <typename Group, typename Iter, size_t Extent>
typename std::enable_if<sycl::is_group_v<std::decay_t<Group>>, void>::type
joint_sort(experimental::group_with_scratchpad<Group, Extent> exec, Iter first,
           Iter last) {
  joint_sort(exec.get_group(), first, last,
             default_sorters::joint_sorter<>(exec.get_memory()));
}

template <typename Group, typename T, typename U, typename Sorter>
// TODO: ADD check for is_sorter , detail::is_sorter<Sorter, Group,
// Iter>::value, void>::type>
std::tuple<T, U> sort_key_value_over_group(Group g, T key, U value,
                                           Sorter sorter) {
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

template <typename GroupHelper, typename T, typename U, typename Compare,
          typename Group, std::size_t Extent>
std::tuple<T, U> sort_key_value_over_group(
    experimental::group_with_scratchpad<Group, Extent> exec, T key, U value,
    Compare comp) {
  return sort_key_value_over_group(
      exec.get_group(), key, value,
      default_sorters::group_key_value_sorter<T, U, Compare>(exec.get_memory(),
                                                             comp));
}

template <typename T, typename U, typename Group, std::size_t Extent>
std::tuple<T, U> sort_key_value_over_group(
    experimental::group_with_scratchpad<Group, Extent> exec, T key, U value) {
  return sort_key_value_over_group(
      exec.get_group(), key, value,
      default_sorters::group_key_value_sorter<T, U>(exec.get_memory()));
}

// key value sorting
template <std::size_t ElementsPerWorkItem, typename Group, typename T,
          typename U, typename ArraySorter,
          typename Property = detail::is_blocked>
void sort_key_value_over_group(Group group,
                               sycl::span<T, ElementsPerWorkItem> keys,
                               sycl::span<U, ElementsPerWorkItem> values,
                               ArraySorter array_sorter,
                               Property property = {}) {
  array_sorter(group, keys, values, property);
}

template <typename Group, typename T, typename U, std::size_t Extent,
          std::size_t ElementsPerWorkItem, typename Compare,
          typename Property = detail::is_blocked>
void sort_key_value_over_group(
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
void sort_key_value_over_group(
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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
#endif
