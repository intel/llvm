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

template <typename Sorter, typename Group, typename Key, typename Value,
          typename = void>
struct is_key_value_sorter : std::false_type {};

template <typename Sorter, typename Group, typename Key, typename Value>
struct is_key_value_sorter<
    Sorter, Group, Key, Value,
    std::enable_if_t<
        std::is_same_v<std::invoke_result_t<Sorter, Group, Key, Value>,
                       std::tuple<Key, Value>> &&
        sycl::is_group_v<Group>>> : std::true_type {};

} // namespace detail

// ---- sort_over_group
template <typename Group, typename T, typename Sorter>
std::enable_if_t<detail::is_sorter<Sorter, Group, T>::value, T>
sort_over_group([[maybe_unused]] Group group, [[maybe_unused]] T value,
                [[maybe_unused]] Sorter sorter) {
#ifdef __SYCL_DEVICE_ONLY__
  return sorter(group, value);
#else
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
      default_sorters::group_sorter<T, Compare, 1>(exec.get_memory(), comp));
}

template <typename Group, typename T, size_t Extent>
std::enable_if_t<sycl::is_group_v<std::decay_t<Group>>, T>
sort_over_group(experimental::group_with_scratchpad<Group, Extent> exec,
                T value) {
  return sort_over_group(exec.get_group(), value,
                         default_sorters::group_sorter<T>(exec.get_memory()));
}

// ---- joint_sort
template <typename Group, typename Iter, typename Sorter>
std::enable_if_t<detail::is_sorter<Sorter, Group, Iter>::value, void>
joint_sort([[maybe_unused]] Group group, [[maybe_unused]] Iter first,
           [[maybe_unused]] Iter last, [[maybe_unused]] Sorter sorter) {
#ifdef __SYCL_DEVICE_ONLY__
  sorter(group, first, last);
#else
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

template <typename Group, typename KeyTy, typename ValueTy, typename Sorter>
std::enable_if_t<
    detail::is_key_value_sorter<Sorter, Group, KeyTy, ValueTy>::value,
    std::tuple<KeyTy, ValueTy>>
sort_key_value_over_group([[maybe_unused]] Group g, [[maybe_unused]] KeyTy key,
                          [[maybe_unused]] ValueTy value,
                          [[maybe_unused]] Sorter sorter) {
#ifdef __SYCL_DEVICE_ONLY__
  return sorter(g, key, value);
#else
  throw sycl::exception(
      std::error_code(PI_ERROR_INVALID_DEVICE, sycl::sycl_category()),
      "Group algorithms are not supported on host device.");
#endif
}

template <typename Group, typename KeyTy, typename ValueTy, typename Compare,
          std::size_t Extent>
std::enable_if_t<
    !detail::is_key_value_sorter<Compare, Group, KeyTy, ValueTy>::value,
    std::tuple<KeyTy, ValueTy>>
sort_key_value_over_group(
    experimental::group_with_scratchpad<Group, Extent> exec, KeyTy key,
    ValueTy value, Compare comp) {
  return sort_key_value_over_group(
      exec.get_group(), key, value,
      default_sorters::group_key_value_sorter<KeyTy, ValueTy, Compare>(
          exec.get_memory(), comp));
}

template <typename KeyTy, typename ValueTy, typename Group, std::size_t Extent>
std::enable_if_t<sycl::is_group_v<std::decay_t<Group>>,
                 std::tuple<KeyTy, ValueTy>>
sort_key_value_over_group(
    experimental::group_with_scratchpad<Group, Extent> exec, KeyTy key,
    ValueTy value) {
  return sort_key_value_over_group(
      exec.get_group(), key, value,
      default_sorters::group_key_value_sorter<KeyTy, ValueTy>(
          exec.get_memory()));
}

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
#endif
