//==------- properties.hpp - SYCL properties associated with kernels -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>                                             // for array
#include <stddef.h>                                          // for size_t
#include <stdint.h>                                          // for uint32_t
#include <sycl/aspects.hpp>                                  // for aspect
#include <sycl/ext/oneapi/experimental/forward_progress.hpp> // for forward_progress_guarantee enum
#include <sycl/ext/oneapi/kernel_properties/function_launch_properties.hpp>
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <type_traits>                                   // for true_type
#include <utility>                                       // for declval
namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

struct properties_tag {};

struct device_has_key
    : detail::compile_time_property_key<detail::PropKind::DeviceHas> {
  template <aspect... Aspects>
  using value_t = property_value<device_has_key,
                                 std::integral_constant<aspect, Aspects>...>;
};

template <aspect... Aspects>
struct property_value<device_has_key,
                      std::integral_constant<aspect, Aspects>...>
    : detail::property_base<
          property_value<device_has_key,
                         std::integral_constant<aspect, Aspects>...>,
          detail::PropKind::DeviceHas, device_has_key> {
  static constexpr std::array<aspect, sizeof...(Aspects)> value{Aspects...};
};

template <aspect... Aspects>
inline constexpr device_has_key::value_t<Aspects...> device_has;

struct work_group_progress_key
    : detail::compile_time_property_key<detail::PropKind::WorkGroupProgress> {
  template <forward_progress_guarantee Guarantee,
            execution_scope CoordinationScope>
  using value_t = property_value<
      work_group_progress_key,
      std::integral_constant<forward_progress_guarantee, Guarantee>,
      std::integral_constant<execution_scope, CoordinationScope>>;
};

struct sub_group_progress_key
    : detail::compile_time_property_key<detail::PropKind::SubGroupProgress> {
  template <forward_progress_guarantee Guarantee,
            execution_scope CoordinationScope>
  using value_t = property_value<
      sub_group_progress_key,
      std::integral_constant<forward_progress_guarantee, Guarantee>,
      std::integral_constant<execution_scope, CoordinationScope>>;
};

struct work_item_progress_key
    : detail::compile_time_property_key<detail::PropKind::WorkItemProgress> {
  template <forward_progress_guarantee Guarantee,
            execution_scope CoordinationScope>
  using value_t = property_value<
      work_item_progress_key,
      std::integral_constant<forward_progress_guarantee, Guarantee>,
      std::integral_constant<execution_scope, CoordinationScope>>;
};

template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
struct property_value<
    work_group_progress_key,
    std::integral_constant<forward_progress_guarantee, Guarantee>,
    std::integral_constant<execution_scope, CoordinationScope>>
    : detail::property_base<
          property_value<
              work_group_progress_key,
              std::integral_constant<forward_progress_guarantee, Guarantee>,
              std::integral_constant<execution_scope, CoordinationScope>>,
          detail::PropKind::WorkGroupProgress, work_group_progress_key> {
  static constexpr forward_progress_guarantee guarantee = Guarantee;
  static constexpr execution_scope coordinationScope = CoordinationScope;
};

template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
struct property_value<
    sub_group_progress_key,
    std::integral_constant<forward_progress_guarantee, Guarantee>,
    std::integral_constant<execution_scope, CoordinationScope>>
    : detail::property_base<
          property_value<
              sub_group_progress_key,
              std::integral_constant<forward_progress_guarantee, Guarantee>,
              std::integral_constant<execution_scope, CoordinationScope>>,
          detail::PropKind::SubGroupProgress, sub_group_progress_key> {
  static constexpr forward_progress_guarantee guarantee = Guarantee;
  static constexpr execution_scope coordinationScope = CoordinationScope;
};

template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
struct property_value<
    work_item_progress_key,
    std::integral_constant<forward_progress_guarantee, Guarantee>,
    std::integral_constant<execution_scope, CoordinationScope>>
    : detail::property_base<
          property_value<
              work_item_progress_key,
              std::integral_constant<forward_progress_guarantee, Guarantee>,
              std::integral_constant<execution_scope, CoordinationScope>>,
          detail::PropKind::WorkItemProgress, work_item_progress_key> {
  static constexpr forward_progress_guarantee guarantee = Guarantee;
  static constexpr execution_scope coordinationScope = CoordinationScope;
};

template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
inline constexpr work_group_progress_key::value_t<Guarantee, CoordinationScope>
    work_group_progress;

template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
inline constexpr sub_group_progress_key::value_t<Guarantee, CoordinationScope>
    sub_group_progress;

template <forward_progress_guarantee Guarantee,
          execution_scope CoordinationScope>
inline constexpr work_item_progress_key::value_t<Guarantee, CoordinationScope>
    work_item_progress;

namespace detail {
template <sycl::aspect... Aspects>
struct HasCompileTimeEffect<device_has_key::value_t<Aspects...>>
    : std::true_type {};
template <aspect... Aspects>
struct PropertyMetaInfo<device_has_key::value_t<Aspects...>> {
  static constexpr const char *name = "sycl-device-has";
  static constexpr const char *value =
      SizeListToStr<static_cast<size_t>(Aspects)...>::value;
};
template <aspect... Aspects>
struct FunctionPropertyMetaInfo<device_has_key::value_t<Aspects...>> {
  static constexpr const char *name = "sycl-device-has";
  static constexpr const char *value =
      SizeListToStr<static_cast<size_t>(Aspects)...>::value;
};

template <typename T, typename = void>
struct HasKernelPropertiesGetMethod : std::false_type {};

template <typename T>
struct HasKernelPropertiesGetMethod<T,
                                    std::void_t<decltype(std::declval<T>().get(
                                        std::declval<properties_tag>()))>>
    : std::true_type {
  using properties_t =
      decltype(std::declval<T>().get(std::declval<properties_tag>()));
};

// If work_group_size and max_work_group_size coexist, check that the
// dimensionality matches and that the required work-group size doesn't
// trivially exceed the maximum size.
template <typename Properties>
struct ConflictingProperties<max_work_group_size_key, Properties> {
  static constexpr bool value = []() constexpr {
    if constexpr (Properties::template has_property<work_group_size_key>()) {
      constexpr auto wg_size =
          Properties::template get_property<work_group_size_key>();
      constexpr auto max_wg_size =
          Properties::template get_property<max_work_group_size_key>();
      static_assert(
          wg_size.size() == max_wg_size.size(),
          "work_group_size and max_work_group_size dimensionality must match");
      if constexpr (wg_size.size() == max_wg_size.size()) {
        constexpr auto Dims = wg_size.size();
        static_assert(Dims < 1 || wg_size[0] <= max_wg_size[0],
                      "work_group_size must not exceed max_work_group_size");
        static_assert(Dims < 2 || wg_size[1] <= max_wg_size[1],
                      "work_group_size must not exceed max_work_group_size");
        static_assert(Dims < 3 || wg_size[2] <= max_wg_size[2],
                      "work_group_size must not exceed max_work_group_size");
      }
    }
    return false;
  }();
};

// If work_group_size and max_linear_work_group_size coexist, check that the
// required linear work-group size doesn't trivially exceed the maximum size.
template <typename Properties>
struct ConflictingProperties<max_linear_work_group_size_key, Properties> {
  static constexpr bool value = []() constexpr {
    if constexpr (Properties::template has_property<work_group_size_key>()) {
      constexpr auto wg_size =
          Properties::template get_property<work_group_size_key>();
      constexpr auto dims = wg_size.size();
      constexpr auto linear_size = wg_size[0] * (dims > 1 ? wg_size[1] : 1) *
                                   (dims > 2 ? wg_size[2] : 1);
      constexpr auto max_linear_wg_size =
          Properties::template get_property<max_linear_work_group_size_key>();
      static_assert(
          linear_size < max_linear_wg_size.value,
          "work_group_size must not exceed max_linear_work_group_size");
    }
    return false;
  }();
};

// If the kernel (last element in the parameter pack) has a get(properties_tag)
// method, return the property list specified by this getter. Otherwise, return
// an empty properety list.
template <typename... RestT>
auto RetrieveGetMethodPropertiesOrEmpty(RestT &&...Rest) {
  // Note: the following trivial identity lambda is used to avoid the issue
  // that line "const auto &KernelObj = (Rest, ...);" may result in a "left
  // operand of comma operator has no effect" error for certain compiler(s)
  auto Identity = [](const auto &x) -> decltype(auto) { return x; };
  const auto &KernelObj = (Identity(Rest), ...);
  if constexpr (ext::oneapi::experimental::detail::HasKernelPropertiesGetMethod<
                    decltype(KernelObj)>::value) {
    return KernelObj.get(ext::oneapi::experimental::properties_tag{});
  } else {
    return ext::oneapi::experimental::empty_properties_t{};
  }
}

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
