//==-------- alloc_base.hpp - SYCL annotated usm basic allocation -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/annotated_arg/annotated_ptr.hpp>
#include <sycl/ext/oneapi/annotated_usm/alloc_util.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

////
//  Parameterized USM allocation functions with properties support
//  These functions take a USM kind parameter that specifies the type of USM to
//  allocate
////

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<void, propertyListA, propertyListB>::value,
                 annotated_ptr<void, propertyListB>>
aligned_alloc_annotated(size_t alignment, size_t numBytes,
                        const device &syclDevice, const context &syclContext,
                        alloc kind,
                        const propertyListA &propList = properties{}) {
  size_t alignFromPropList = GetAlignFromPropList<propertyListA>::value;
  const property_list &usmPropList = get_usm_property_list(propList);

  if constexpr (HasUsmKind<propertyListA>::value) {
    constexpr alloc usmKind = GetUsmKindFromPropList<propertyListA>::value;
    if (usmKind != kind) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::invalid),
          "Input property list of USM allocation function contains usm_kind "
          "property that conflicts with the usm kind argument");
    }
  }

  void *rawPtr = nullptr;
  switch (kind) {
  case alloc::device:
    rawPtr = sycl::aligned_alloc_device(
        combine_align(alignment, alignFromPropList), numBytes, syclDevice,
        syclContext, usmPropList);
    check_device_aspect(syclDevice);
    break;
  case alloc::host:
    rawPtr =
        sycl::aligned_alloc_host(combine_align(alignment, alignFromPropList),
                                 numBytes, syclContext, usmPropList);
    check_host_aspect(syclContext);
    break;
  case alloc::shared:
    rawPtr = sycl::aligned_alloc_shared(
        combine_align(alignment, alignFromPropList), numBytes, syclDevice,
        syclContext, usmPropList);
    check_shared_aspect(syclDevice);
    break;
  default:
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Unknown USM kind allocation function");
  }
  return {rawPtr};
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<T, propertyListA, propertyListB>::value,
                 annotated_ptr<T, propertyListB>>
aligned_alloc_annotated(size_t alignment, size_t count,
                        const device &syclDevice, const context &syclContext,
                        alloc kind,
                        const propertyListA &propList = properties{}) {
  return {static_cast<T *>(aligned_alloc_annotated(alignment, count * sizeof(T), syclDevice,
                                       syclContext, kind, propList)
              .get())};
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<void, propertyListA, propertyListB>::value,
                 annotated_ptr<void, propertyListB>>
aligned_alloc_annotated(size_t alignment, size_t numBytes,
                        const queue &syclQueue, alloc kind,
                        const propertyListA &propList = properties{}) {
  return aligned_alloc_annotated(alignment, numBytes, syclQueue.get_device(),
                                 syclQueue.get_context(), kind, propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<T, propertyListA, propertyListB>::value,
                 annotated_ptr<T, propertyListB>>
aligned_alloc_annotated(size_t alignment, size_t count, const queue &syclQueue,
                        alloc kind,
                        const propertyListA &propList = properties{}) {
  return aligned_alloc_annotated<T>(alignment, count, syclQueue.get_device(),
                                    syclQueue.get_context(), kind, propList);
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<void, propertyListA, propertyListB>::value,
                 annotated_ptr<void, propertyListB>>
malloc_annotated(size_t numBytes, const device &syclDevice,
                 const context &syclContext, alloc kind,
                 const propertyListA &propList = properties{}) {
  return aligned_alloc_annotated(0, numBytes, syclDevice, syclContext, kind,
                                 propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<T, propertyListA, propertyListB>::value,
                 annotated_ptr<T, propertyListB>>
malloc_annotated(size_t count, const device &syclDevice,
                 const context &syclContext, alloc kind,
                 const propertyListA &propList = properties{}) {
  return {static_cast<T *>(malloc_annotated(count * sizeof(T), syclDevice, syclContext,
                                kind, propList)
              .get())};
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<void, propertyListA, propertyListB>::value,
                 annotated_ptr<void, propertyListB>>
malloc_annotated(size_t numBytes, const queue &syclQueue, alloc kind,
                 const propertyListA &propList = properties{}) {
  return malloc_annotated(numBytes, syclQueue.get_device(),
                          syclQueue.get_context(), kind, propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<T, propertyListA, propertyListB>::value,
                 annotated_ptr<T, propertyListB>>
malloc_annotated(size_t count, const queue &syclQueue, alloc kind,
                 const propertyListA &propList = properties{}) {
  return malloc_annotated<T>(count, syclQueue.get_device(),
                             syclQueue.get_context(), kind, propList);
}

////
//  Additional USM memory allocation functions with properties support that
//  requires the usm_kind property
////

template <typename propertyListA,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<void, propertyListA, propertyListB>::value,
                 annotated_ptr<void, propertyListB>>
malloc_annotated(size_t numBytes, const device &syclDevice,
                 const context &syclContext, const propertyListA &propList) {

  constexpr alloc usmKind = GetUsmKindFromPropList<propertyListA>::value;
  static_assert(usmKind != alloc::unknown,
                "USM kind is missing in the input property list.");
  return malloc_annotated(numBytes, syclDevice, syclContext, usmKind, propList);
}

template <typename T, typename propertyListA,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<T, propertyListA, propertyListB>::value,
                 annotated_ptr<T, propertyListB>>
malloc_annotated(size_t count, const device &syclDevice,
                 const context &syclContext, const propertyListA &propList) {
  return {static_cast<T *>(malloc_annotated(count * sizeof(T), syclDevice, syclContext,
                                propList)
              .get())};
}

template <typename propertyListA,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<void, propertyListA, propertyListB>::value,
                 annotated_ptr<void, propertyListB>>
malloc_annotated(size_t numBytes, const queue &syclQueue,
                 const propertyListA &propList) {
  return malloc_annotated(numBytes, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA,
          typename propertyListB =
              typename GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<CheckTAndPropLists<T, propertyListA, propertyListB>::value,
                 annotated_ptr<T, propertyListB>>
malloc_annotated(size_t count, const queue &syclQueue,
                 const propertyListA &propList) {
  return malloc_annotated<T>(count, syclQueue.get_device(),
                             syclQueue.get_context(), propList);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl