//==-------- alloc_device.hpp - SYCL annotated usm device allocation -------==//
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
using CheckDevicePtrTAndPropLists =
    CheckTAndPropListsWithUsmKind<alloc::device, T, ListA, ListB>;

template <typename PropertyListT>
using GetAnnotatedDevicePtrProperties =
    GetAnnotatedPtrPropertiesWithUsmKind<alloc::device, PropertyListT>;

////
//  Aligned device USM allocation functions with properties support
////

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
aligned_alloc_device_annotated(size_t alignment, size_t numBytes,
                               const device &syclDevice,
                               const context &syclContext,
                               const propertyListA &propList = properties{}) {
  return aligned_alloc_annotated<propertyListB>(alignment, numBytes, syclDevice,
                                                syclContext, alloc::device);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
aligned_alloc_device_annotated(size_t alignment, size_t count,
                               const device &syclDevice,
                               const context &syclContext,
                               const propertyListA &propList = properties{}) {
  return {static_cast<T *>(aligned_alloc_device_annotated(alignment, count * sizeof(T),
                                              syclDevice, syclContext, propList)
              .get())};
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
aligned_alloc_device_annotated(size_t alignment, size_t numBytes,
                               const queue &syclQueue,
                               const propertyListA &propList = properties{}) {
  return aligned_alloc_device_annotated(alignment, numBytes,
                                        syclQueue.get_device(),
                                        syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
aligned_alloc_device_annotated(size_t alignment, size_t count,
                               const queue &syclQueue,
                               const propertyListA &propList = properties{}) {
  return aligned_alloc_device_annotated<T>(alignment, count,
                                           syclQueue.get_device(),
                                           syclQueue.get_context(), propList);
}

////
//  Device USM allocation functions with properties support
////

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_device_annotated(size_t numBytes, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = properties{}) {
  return aligned_alloc_device_annotated(0, numBytes, syclDevice, syclContext,
                                        propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_device_annotated(size_t count, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = properties{}) {
  return {static_cast<T *>(malloc_device_annotated(count * sizeof(T), syclDevice,
                                       syclContext, propList)
              .get())};
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_device_annotated(size_t numBytes, const queue &syclQueue,
                        const propertyListA &propList = properties{}) {
  return malloc_device_annotated(numBytes, syclQueue.get_device(),
                                 syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetAnnotatedDevicePtrProperties<propertyListA>::type>
std::enable_if_t<
    CheckDevicePtrTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_device_annotated(size_t count, const queue &syclQueue,
                        const propertyListA &propList = properties{}) {
  return malloc_device_annotated<T>(count, syclQueue.get_device(),
                                    syclQueue.get_context(), propList);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl