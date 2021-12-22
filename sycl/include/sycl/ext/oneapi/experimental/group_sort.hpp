//==--------- group_sort.hpp --- SYCL extension group sorting algorithm-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
#include <CL/sycl/detail/defines_elementary.hpp>
#include <CL/sycl/detail/group_sort_impl.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <type_traits>

#include "group_helpers_sorters.hpp"

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental {
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
  static decltype(
      std::integral_constant<bool, is_expected_return_type<G>::value &&
                                       sycl::is_group_v<G>>{}) test(int);

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
                  sycl::detail::is_generic_group<G>{}) test(int);

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
      std::error_code(PI_INVALID_DEVICE, sycl::sycl_category()),
      "Group algorithms are not supported on host device.");
#endif
}

template <typename Group, typename T, typename Compare, std::size_t Extent>
typename std::enable_if<!detail::is_sorter<Compare, Group, T>::value, T>::type
sort_over_group(experimental::group_with_scratchpad<Group, Extent> exec,
                T value, Compare comp) {
  return sort_over_group(
      exec.get_group(), value,
      experimental::default_sorter<Compare>(exec.get_memory(), comp));
}

template <typename Group, typename T, std::size_t Extent>
typename std::enable_if<sycl::is_group_v<std::decay_t<Group>>, T>::type
sort_over_group(experimental::group_with_scratchpad<Group, Extent> exec,
                T value) {
  return sort_over_group(exec.get_group(), value,
                         experimental::default_sorter<>(exec.get_memory()));
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
      std::error_code(PI_INVALID_DEVICE, sycl::sycl_category()),
      "Group algorithms are not supported on host device.");
#endif
}

template <typename Group, typename Iter, typename Compare, std::size_t Extent>
typename std::enable_if<!detail::is_sorter<Compare, Group, Iter>::value,
                        void>::type
joint_sort(experimental::group_with_scratchpad<Group, Extent> exec, Iter first,
           Iter last, Compare comp) {
  joint_sort(exec.get_group(), first, last,
             experimental::default_sorter<Compare>(exec.get_memory(), comp));
}

template <typename Group, typename Iter, std::size_t Extent>
typename std::enable_if<sycl::is_group_v<std::decay_t<Group>>, void>::type
joint_sort(experimental::group_with_scratchpad<Group, Extent> exec, Iter first,
           Iter last) {
  joint_sort(exec.get_group(), first, last,
             experimental::default_sorter<>(exec.get_memory()));
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
#endif // __cplusplus >=201703L
