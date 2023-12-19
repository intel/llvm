//==-------- alloc_host.hpp - SYCL annotated usm host allocation -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/experimental/annotated_usm/alloc_base.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

template <typename T, typename ListA, typename ListB>
using CheckHostPtrTAndPropLists =
    typename detail::CheckTAndPropListsWithUsmKind<sycl::usm::alloc::host, T,
                                                   ListA, ListB>;

template <typename PropertyListT>
using GetAnnotatedHostPtrProperties =
    detail::GetAnnotatedPtrPropertiesWithUsmKind<sycl::usm::alloc::host,
                                                 PropertyListT>;

////
//  "aligned_alloc_host_annotated": Aligned host USM allocation functions with
//  properties support
//
//  This the base form of all the annotated USM host allocation functions, which
//  are implemented by calling the more generic "aligned_alloc_annotated"
//  functions with the USM kind as an argument. Note that when calling
//  "aligned_alloc_annotated", the template parameter `propertyListA` should
//  include the `usm_kind<alloc::host>` property to make it appear on the
//  returned annotated_ptr of "aligned_alloc_annotated"
////

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedHostPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckHostPtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
aligned_alloc_host_annotated(size_t alignment, size_t numBytes,
                             const context &syclContext,
                             const propertyListA &propList = properties{}) {
  auto tmp = aligned_alloc_annotated(alignment, numBytes, {}, syclContext,
                                     sycl::usm::alloc::host, propList);
  return annotated_ptr<void, propertyListB>(tmp.get());
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedHostPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckHostPtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
aligned_alloc_host_annotated(size_t alignment, size_t count,
                             const context &syclContext,
                             const propertyListA &propList = properties{}) {
  auto tmp = aligned_alloc_annotated<T>(alignment, count, {}, syclContext,
                                        sycl::usm::alloc::host, propList);
  return annotated_ptr<T, propertyListB>(tmp.get());
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedHostPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckHostPtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
aligned_alloc_host_annotated(size_t alignment, size_t numBytes,
                             const queue &syclQueue,
                             const propertyListA &propList = properties{}) {
  return aligned_alloc_host_annotated(alignment, numBytes,
                                      syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedHostPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckHostPtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
aligned_alloc_host_annotated(size_t alignment, size_t count,
                             const queue &syclQueue,
                             const propertyListA &propList = properties{}) {
  return aligned_alloc_host_annotated<T>(alignment, count,
                                         syclQueue.get_context(), propList);
}

////
//  Host USM allocation functions with properties support
//
//  Note: "malloc_host_annotated" functions call "aligned_alloc_host_annotated"
//  with alignment 0
////

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedHostPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckHostPtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_host_annotated(size_t numBytes, const context &syclContext,
                      const propertyListA &propList = properties{}) {
  return aligned_alloc_host_annotated(0, numBytes, syclContext, propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedHostPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckHostPtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_host_annotated(size_t count, const context &syclContext,
                      const propertyListA &propList = properties{}) {
  return aligned_alloc_host_annotated<T>(0, count, syclContext, propList);
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedHostPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckHostPtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_host_annotated(size_t numBytes, const queue &syclQueue,
                      const propertyListA &propList = properties{}) {
  return malloc_host_annotated(numBytes, syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedHostPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckHostPtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_host_annotated(size_t count, const queue &syclQueue,
                      const propertyListA &propList = properties{}) {
  return malloc_host_annotated<T>(count, syclQueue.get_context(), propList);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl