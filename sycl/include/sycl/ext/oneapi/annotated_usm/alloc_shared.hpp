//==-------- alloc_shared.hpp - SYCL annotated usm shared allocation -------==//
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
using CheckSharedPtrTAndPropLists =
    CheckTAndPropListsWithUsmKind<alloc::shared, T, ListA, ListB>;

template <typename PropertyListT>
using GetAnnotatedSharedPtrProperties =
    GetAnnotatedPtrPropertiesWithUsmKind<alloc::shared, PropertyListT>;

////
//  Aligned shared USM allocation functions with properties support
////

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedSharedPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckSharedPtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
aligned_alloc_shared_annotated(size_t alignment, size_t numBytes,
                               const device &syclDevice,
                               const context &syclContext,
                               const propertyListA &propList = properties{}) {

  return aligned_alloc_annotated<propertyListB>(alignment, numBytes, syclDevice,
                                                syclContext, alloc::shared);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedSharedPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckSharedPtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
aligned_alloc_shared_annotated(size_t alignment, size_t count,
                               const device &syclDevice,
                               const context &syclContext,
                               const propertyListA &propList = properties{}) {
  return {static_cast<T *>(aligned_alloc_shared_annotated(alignment, count * sizeof(T),
                                              syclDevice, syclContext, propList)
              .get())};
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedSharedPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckSharedPtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
aligned_alloc_shared_annotated(size_t alignment, size_t numBytes,
                               const queue &syclQueue,
                               const propertyListA &propList = properties{}) {
  return aligned_alloc_shared_annotated(alignment, numBytes,
                                        syclQueue.get_device(),
                                        syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedSharedPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckSharedPtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
aligned_alloc_shared_annotated(size_t alignment, size_t count,
                               const queue &syclQueue,
                               const propertyListA &propList = properties{}) {
  return aligned_alloc_shared_annotated<T>(alignment, count,
                                           syclQueue.get_device(),
                                           syclQueue.get_context(), propList);
}

////
//  Shared USM allocation functions with properties support
////

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedSharedPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckSharedPtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_shared_annotated(size_t numBytes, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = properties{}) {
  return aligned_alloc_shared_annotated(0, numBytes, syclDevice, syclContext,
                                        propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedSharedPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckSharedPtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_shared_annotated(size_t count, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = properties{}) {
  return {static_cast<T *>(malloc_shared_annotated(count * sizeof(T), syclDevice,
                                       syclContext, propList)
              .get())};
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedSharedPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckSharedPtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_shared_annotated(size_t numBytes, const queue &syclQueue,
                        const propertyListA &propList = properties{}) {
  return malloc_shared_annotated(numBytes, syclQueue.get_device(),
                                 syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedSharedPtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckSharedPtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_shared_annotated(size_t count, const queue &syclQueue,
                        const propertyListA &propList = properties{}) {
  return malloc_shared_annotated<T>(count, syclQueue.get_device(),
                                    syclQueue.get_context(), propList);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl