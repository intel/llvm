//==---- free_function_queries.hpp -- SYCL_INTEL_free_function_queries ext -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/builder.hpp>
#include <sycl/detail/nd_item_core.hpp>
#include <sycl/detail/sub_group_core.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
template <int Dimensions> item<Dimensions, true> getFreeFunctionQueryItem() {
  static_assert(Dimensions > 0 && Dimensions < 4, "invalid dimensions");
#ifdef __SYCL_DEVICE_ONLY__
  id<Dimensions> GlobalId{
      __spirv::initBuiltInGlobalInvocationId<Dimensions, id<Dimensions>>()};
  range<Dimensions> GlobalSize{
      __spirv::initBuiltInGlobalSize<Dimensions, range<Dimensions>>()};
  id<Dimensions> GlobalOffset{
      __spirv::initBuiltInGlobalOffset<Dimensions, id<Dimensions>>()};
  return item<Dimensions, true>(GlobalSize, GlobalId, GlobalOffset);
#else
  return item<Dimensions, true>(range<Dimensions>{}, id<Dimensions>{},
                                id<Dimensions>{});
#endif
}
} // namespace detail

namespace ext::oneapi::this_work_item {
template <int Dimensions> nd_item<Dimensions> get_nd_item() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::Builder::getNDItem<Dimensions>();
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}

template <int Dimensions> group<Dimensions> get_work_group() {
  return get_nd_item<Dimensions>().get_group();
}

inline sycl::sub_group get_sub_group() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::sub_group();
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}
} // namespace ext::oneapi::this_work_item

namespace ext::oneapi::experimental {
template <int Dims>
__SYCL_DEPRECATED(
    "use sycl::ext::oneapi::this_work_item::get_nd_item() instead")
nd_item<Dims> this_nd_item() {
  return ext::oneapi::this_work_item::get_nd_item<Dims>();
}

template <int Dims>
__SYCL_DEPRECATED(
    "use sycl::ext::oneapi::this_work_item::get_work_group() instead")
group<Dims> this_group() {
  return ext::oneapi::this_work_item::get_work_group<Dims>();
}

__SYCL_DEPRECATED(
    "use sycl::ext::oneapi::this_work_item::get_sub_group() instead")
inline sycl::sub_group this_sub_group() {
  return ext::oneapi::this_work_item::get_sub_group();
}

template <int Dims>
__SYCL_DEPRECATED("use nd_range kernel and "
                  "sycl::ext::oneapi::this_work_item::get_nd_item() instead")
item<Dims, true> this_item() {
#ifdef __SYCL_DEVICE_ONLY__
  return sycl::detail::getFreeFunctionQueryItem<Dims>();
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}

template <int Dims>
__SYCL_DEPRECATED("use nd_range kernel and "
                  "sycl::ext::oneapi::this_work_item::get_nd_item() instead")
id<Dims> this_id() {
#ifdef __SYCL_DEVICE_ONLY__
  static_assert(Dims > 0 && Dims < 4, "invalid dimensions");
  return __spirv::initBuiltInGlobalInvocationId<Dims, id<Dims>>();
#else
  throw sycl::exception(
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Free function calls are not supported on host");
#endif
}
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
