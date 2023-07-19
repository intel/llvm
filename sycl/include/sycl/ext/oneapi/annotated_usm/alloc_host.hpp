//==-------- alloc_host.hpp - SYCL annotated usm host allocation -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/annotated_usm/alloc_base.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

template <typename T, typename ListA, typename ListB>
using CheckHostPtrTAndPropLists =
    CheckTAndPropListsWithUsmKind<alloc::host, T, ListA, ListB>;

template <typename PropertyListT>
using GetAnnotatedHostPtrProperties =
    GetAnnotatedPtrPropertiesWithUsmKind<alloc::host, PropertyListT>;

////
//  Aligned host USM allocation functions with properties support
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

  return aligned_alloc_annotated<propertyListB>(alignment, numBytes, {},
                                                syclContext, alloc::host);
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
  return {static_cast<T *>(aligned_alloc_host_annotated(alignment, count * sizeof(T),
                                            syclContext, propList)
              .get())};
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
  return {static_cast<T *>(malloc_host_annotated(count * sizeof(T), syclContext, propList)
              .get())};
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
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl