//==--- function_properties.hpp - SYCL standalone function properties -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// This header is the lightweight split entry point for free-function kernel
// annotations. Keep the public property_value definitions here, rather than
// only in kernel_properties/properties.hpp, so standalone users and umbrella
// users observe the same decltype(...) while avoiding the heavier property-list
// machinery on this path.

#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename PropertyT> struct FunctionPropertyMetaInfo;

} // namespace detail

struct nd_range_kernel_key
    : detail::compile_time_property_key<detail::PropKind::NDRangeKernel> {
  template <int Dims>
  using value_t =
      property_value<nd_range_kernel_key, std::integral_constant<int, Dims>>;
};

struct single_task_kernel_key
    : detail::compile_time_property_key<detail::PropKind::SingleTaskKernel> {
  using value_t = property_value<single_task_kernel_key>;
};

template <int Dims>
struct property_value<nd_range_kernel_key, std::integral_constant<int, Dims>>
    : detail::property_base<property_value<nd_range_kernel_key,
                                           std::integral_constant<int, Dims>>,
                            detail::PropKind::NDRangeKernel,
                            nd_range_kernel_key> {
  static_assert(Dims >= 1 && Dims <= 3,
                "nd_range_kernel must use dimension 1, 2, or 3.");

  using value_t = int;
  static constexpr int dimensions = Dims;
};

template <>
struct property_value<single_task_kernel_key>
    : detail::property_base<property_value<single_task_kernel_key>,
                            detail::PropKind::SingleTaskKernel,
                            single_task_kernel_key> {};

template <int Dims>
inline constexpr nd_range_kernel_key::value_t<Dims> nd_range_kernel;

inline constexpr single_task_kernel_key::value_t single_task_kernel;

namespace detail {

template <int Dims>
struct FunctionPropertyMetaInfo<nd_range_kernel_key::value_t<Dims>> {
  static constexpr const char *name = "sycl-nd-range-kernel";
  static constexpr int value = Dims;
};

template <> struct FunctionPropertyMetaInfo<single_task_kernel_key::value_t> {
  static constexpr const char *name = "sycl-single-task-kernel";
  static constexpr int value = 0;
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(PROP)                                \
  [[__sycl_detail__::add_ir_attributes_function(                               \
      sycl::ext::oneapi::experimental::detail::FunctionPropertyMetaInfo<       \
          sycl::ext::oneapi::experimental::detail::remove_cvref_t<             \
              decltype(PROP)>>::name,                                          \
      sycl::ext::oneapi::experimental::detail::FunctionPropertyMetaInfo<       \
          sycl::ext::oneapi::experimental::detail::remove_cvref_t<             \
              decltype(PROP)>>::value)]]
#else
#define SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(PROP)
#endif