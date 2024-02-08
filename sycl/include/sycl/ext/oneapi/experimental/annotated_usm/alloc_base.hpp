//==-------- alloc_base.hpp - SYCL annotated usm basic allocation -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <sycl/ext/oneapi/experimental/annotated_usm/alloc_util.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental {

////
//  Parameterized USM allocation functions with properties support
//  These functions take a USM kind parameter that specifies the type of USM to
//  allocate
//
//  Note: this function group is the base implementation of the other annotated
//  allocation API, and is eventally called by:
//  1. "xxx_alloc_annotated" with USM kind in the properties (defined in
//  alloc_base.hpp)
//  2. "xxx_alloc_device_annotated" (defined in alloc_device.hpp)
//  3. "xxx_alloc_host_annotated"   (defined in alloc_host.hpp)
//  4. "xxx_alloc_shared_annotated" (defined in alloc_shared.hpp)
////
template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
aligned_alloc_annotated(size_t alignment, size_t numBytes,
                        const device &syclDevice, const context &syclContext,
                        sycl::usm::alloc kind,
                        const propertyListA &propList = properties{}) {
  detail::ValidAllocPropertyList<void, propertyListA>::value;

  // The input argument `propList` is useful when propertyListA contains valid
  // runtime properties. While such case is not defined yet, suppress unused
  // variables warning
  static_cast<void>(propList);

  constexpr size_t alignFromPropList =
      detail::GetAlignFromPropList<propertyListA>::value;
  const property_list &usmPropList = get_usm_property_list<propertyListA>();

  if constexpr (detail::HasUsmKind<propertyListA>::value) {
    constexpr sycl::usm::alloc usmKind =
        detail::GetUsmKindFromPropList<propertyListA>::value;
    if (usmKind != kind) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::invalid),
          "Input property list of USM allocation function contains usm_kind "
          "property that conflicts with the usm kind argument");
    }
  }

  if (kind == sycl::usm::alloc::unknown)
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Unknown USM allocation kind was specified.");

  void *rawPtr =
      sycl::aligned_alloc(combine_align(alignment, alignFromPropList), numBytes,
                          syclDevice, syclContext, kind, usmPropList);
  return annotated_ptr<void, propertyListB>(rawPtr);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
aligned_alloc_annotated(size_t alignment, size_t count,
                        const device &syclDevice, const context &syclContext,
                        sycl::usm::alloc kind,
                        const propertyListA &propList = properties{}) {
  detail::ValidAllocPropertyList<T, propertyListA>::value;

  // The input argument `propList` is useful when propertyListA contains valid
  // runtime properties. While such case is not defined yet, suppress unused
  // variables warning
  static_cast<void>(propList);

  constexpr size_t alignFromPropList =
      detail::GetAlignFromPropList<propertyListA>::value;
  const property_list &usmPropList = get_usm_property_list<propertyListA>();

  if constexpr (detail::HasUsmKind<propertyListA>::value) {
    constexpr sycl::usm::alloc usmKind =
        detail::GetUsmKindFromPropList<propertyListA>::value;
    if (usmKind != kind) {
      throw sycl::exception(
          sycl::make_error_code(sycl::errc::invalid),
          "Input property list of USM allocation function contains usm_kind "
          "property that conflicts with the usm kind argument");
    }
  }

  if (kind == sycl::usm::alloc::unknown)
    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                          "Unknown USM allocation kind was specified.");

  size_t combinedAlign = combine_align(alignment, alignFromPropList);
  T *rawPtr = sycl::aligned_alloc<T>(combinedAlign, count, syclDevice,
                                     syclContext, kind, usmPropList);
  return annotated_ptr<T, propertyListB>(rawPtr);
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
aligned_alloc_annotated(size_t alignment, size_t numBytes,
                        const queue &syclQueue, sycl::usm::alloc kind,
                        const propertyListA &propList = properties{}) {
  return aligned_alloc_annotated(alignment, numBytes, syclQueue.get_device(),
                                 syclQueue.get_context(), kind, propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
aligned_alloc_annotated(size_t alignment, size_t count, const queue &syclQueue,
                        sycl::usm::alloc kind,
                        const propertyListA &propList = properties{}) {
  return aligned_alloc_annotated<T>(alignment, count, syclQueue.get_device(),
                                    syclQueue.get_context(), kind, propList);
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_annotated(size_t numBytes, const device &syclDevice,
                 const context &syclContext, sycl::usm::alloc kind,
                 const propertyListA &propList = properties{}) {
  return aligned_alloc_annotated(0, numBytes, syclDevice, syclContext, kind,
                                 propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_annotated(size_t count, const device &syclDevice,
                 const context &syclContext, sycl::usm::alloc kind,
                 const propertyListA &propList = properties{}) {
  return aligned_alloc_annotated<T>(0, count, syclDevice, syclContext, kind,
                                    propList);
}

template <typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_annotated(size_t numBytes, const queue &syclQueue, sycl::usm::alloc kind,
                 const propertyListA &propList = properties{}) {
  return malloc_annotated(numBytes, syclQueue.get_device(),
                          syclQueue.get_context(), kind, propList);
}

template <typename T, typename propertyListA = detail::empty_properties_t,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_annotated(size_t count, const queue &syclQueue, sycl::usm::alloc kind,
                 const propertyListA &propList = properties{}) {
  return malloc_annotated<T>(count, syclQueue.get_device(),
                             syclQueue.get_context(), kind, propList);
}

////
//  Additional USM memory allocation functions with properties support that
//  requires the usm_kind property to be specified on the input property list
//
//  These functions are implemented by extracting the usm kind from the property
//  list and calling the usm-kind-as-argument version
////

template <typename propertyListA,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_annotated(size_t numBytes, const device &syclDevice,
                 const context &syclContext, const propertyListA &propList) {
  constexpr sycl::usm::alloc usmKind =
      detail::GetUsmKindFromPropList<propertyListA>::value;
  static_assert(usmKind != sycl::usm::alloc::unknown,
                "USM kind is not specified. Please specify it as an argument "
                "or in the input property list.");
  return malloc_annotated(numBytes, syclDevice, syclContext, usmKind, propList);
}

template <typename T, typename propertyListA,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_annotated(size_t count, const device &syclDevice,
                 const context &syclContext, const propertyListA &propList) {
  constexpr sycl::usm::alloc usmKind =
      detail::GetUsmKindFromPropList<propertyListA>::value;
  static_assert(usmKind != sycl::usm::alloc::unknown,
                "USM kind is not specified. Please specify it as an argument "
                "or in the input property list.");
  return malloc_annotated<T>(count, syclDevice, syclContext, usmKind, propList);
}

template <typename propertyListA,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<void, propertyListA, propertyListB>::value,
    annotated_ptr<void, propertyListB>>
malloc_annotated(size_t numBytes, const queue &syclQueue,
                 const propertyListA &propList) {
  return malloc_annotated(numBytes, syclQueue.get_device(),
                          syclQueue.get_context(), propList);
}

template <typename T, typename propertyListA,
          typename propertyListB =
              typename detail::GetCompileTimeProperties<propertyListA>::type>
std::enable_if_t<
    detail::CheckTAndPropLists<T, propertyListA, propertyListB>::value,
    annotated_ptr<T, propertyListB>>
malloc_annotated(size_t count, const queue &syclQueue,
                 const propertyListA &propList) {
  return malloc_annotated<T>(count, syclQueue.get_device(),
                             syclQueue.get_context(), propList);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl