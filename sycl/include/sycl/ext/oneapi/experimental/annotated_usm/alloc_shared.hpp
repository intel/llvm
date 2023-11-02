//==-------- alloc_shared.hpp - SYCL annotated usm shared allocation -------==//
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
using CheckSharedPtrTAndPropLists =
    typename detail::CheckTAndPropListsWithUsmKind<sycl::usm::alloc::shared, T,
                                                   ListA, ListB>;

template <typename PropertyListT>
using GetAnnotatedSharedPtrProperties =
    detail::GetAnnotatedPtrPropertiesWithUsmKind<sycl::usm::alloc::shared,
                                                 PropertyListT>;

////
//  Aligned shared USM allocation functions with properties support
//
//  This the base form of all the annotated USM shared allocation functions,
//  which are implemented by calling the more generic "aligned_alloc_annotated"
//  functions with the USM kind as an argument. Note that when calling
//  "aligned_alloc_annotated", the template parameter `propertyListA` should
//  include the `usm_kind<alloc::shared>` property to make it appear on the
//  returned annotated_ptr of "aligned_alloc_annotated"
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
  auto tmp =
      aligned_alloc_annotated(alignment, numBytes, syclDevice, syclContext,
                              sycl::usm::alloc::shared, propList);
  return annotated_ptr<void, propertyListB>(tmp.get());
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
  auto tmp =
      aligned_alloc_annotated<T>(alignment, count, syclDevice, syclContext,
                                 sycl::usm::alloc::shared, propList);
  return annotated_ptr<T, propertyListB>(tmp.get());
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
//
//  Note: "malloc_shared_annotated" functions call
//  "aligned_alloc_shared_annotated" with alignment 0
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
  return aligned_alloc_shared_annotated<T>(0, count, syclDevice, syclContext,
                                           propList);
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
} // namespace _V1
} // namespace sycl