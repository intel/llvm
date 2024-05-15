//==-------- alloc_device.hpp - SYCL annotated usm device allocation -------==//
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
using CheckDevicePtrTAndPropLists =
    typename detail::CheckTAndPropListsWithUsmKind<sycl::usm::alloc::device, T,
                                                   ListA, ListB>;

template <typename PropertyListT>
using GetAnnotatedDevicePtrProperties =
    detail::GetAnnotatedPtrPropertiesWithUsmKind<sycl::usm::alloc::device,
                                                 PropertyListT>;

////
//  "aligned_alloc_device_annotated": aligned device USM allocation functions
//  with properties support
//
//  This the base form of all the annotated USM device allocation functions,
//  which are implemented by calling the more generic "aligned_alloc_annotated"
//  functions with the USM kind as an argument. Note that the returned
//  annotated_ptr of "aligned_alloc_annotated" may not contain  the
//  `usm_kind<alloc::device>`, so reconstruct the real annotated_ptr that
//  contains usm_kind using the raw pointer of "aligned_alloc_annotated" result
////
template <typename propertyListA = empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
aligned_alloc_device_annotated(
    size_t alignment, size_t numBytes, const device &syclDevice,
    const context &syclContext,
    const propertyListA &propList = propertyListA{}) {
  auto tmp =
      aligned_alloc_annotated(alignment, numBytes, syclDevice, syclContext,
                              sycl::usm::alloc::device, propList);
  return annotated_ptr<void, propertyListB>(tmp.get());
}

template <typename T, typename propertyListA = empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
aligned_alloc_device_annotated(
    size_t alignment, size_t count, const device &syclDevice,
    const context &syclContext,
    const propertyListA &propList = propertyListA{}) {
  auto tmp =
      aligned_alloc_annotated<T>(alignment, count, syclDevice, syclContext,
                                 sycl::usm::alloc::device, propList);
  return annotated_ptr<T, propertyListB>(tmp.get());
}

template <typename propertyListA = empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
aligned_alloc_device_annotated(
    size_t alignment, size_t numBytes, const queue &syclQueue,
    const propertyListA &propList = propertyListA{}) {
  return aligned_alloc_device_annotated(alignment, numBytes,
                                        syclQueue.get_device(),
                                        syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA = empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
aligned_alloc_device_annotated(
    size_t alignment, size_t count, const queue &syclQueue,
    const propertyListA &propList = propertyListA{}) {
  return aligned_alloc_device_annotated<T>(alignment, count,
                                           syclQueue.get_device(),
                                           syclQueue.get_context(), propList);
}

////
//  "malloc_device_annotated": device USM allocation functions with properties
//  support
//
//  Note: "malloc_device_annotated" functions call
//  "aligned_alloc_device_annotated" with alignment 0
////
template <typename propertyListA = empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_device_annotated(size_t numBytes, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = propertyListA{}) {
  return aligned_alloc_device_annotated(0, numBytes, syclDevice, syclContext,
                                        propList);
}

template <typename T, typename propertyListA = empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_device_annotated(size_t count, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = propertyListA{}) {
  return aligned_alloc_device_annotated<T>(0, count, syclDevice, syclContext,
                                           propList);
}

template <typename propertyListA = empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_device_annotated(size_t numBytes, const queue &syclQueue,
                        const propertyListA &propList = propertyListA{}) {
  return malloc_device_annotated(numBytes, syclQueue.get_device(),
                                 syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA = empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_device_annotated(size_t count, const queue &syclQueue,
                        const propertyListA &propList = propertyListA{}) {
  return malloc_device_annotated<T>(count, syclQueue.get_device(),
                                    syclQueue.get_context(), propList);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl