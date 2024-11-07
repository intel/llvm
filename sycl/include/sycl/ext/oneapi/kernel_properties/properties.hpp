//==------- properties.hpp - SYCL properties associated with kernels -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array>                                             // for array
#include <limits>
#include <stddef.h>                                          // for size_t
#include <stdint.h>                                          // for uint32_T
#include <sycl/aspects.hpp>                                  // for aspect
#include <sycl/ext/oneapi/experimental/forward_progress.hpp> // for forward_progress_guarantee enum
#include <sycl/ext/oneapi/properties/properties.hpp>
#include <type_traits>                                   // for true_type
#include <utility>                                       // for declval
namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {
// Trait for checking that all size_t values are non-zero.
template <size_t... Xs> struct AllNonZero {
  static constexpr bool value = true;
};
template <size_t X, size_t... Xs> struct AllNonZero<X, Xs...> {
  static constexpr bool value = X > 0 && AllNonZero<Xs...>::value;
};
} // namespace detail

struct properties_tag {};

struct work_group_size_key
    : detail::compile_time_property_key<detail::PropKind::WorkGroupSize> {
  template <size_t... Dims>
  using value_t = property_value<work_group_size_key,
                                 std::integral_constant<size_t, Dims>...>;
};

struct work_group_size_hint_key
    : detail::compile_time_property_key<detail::PropKind::WorkGroupSizeHint> {
  template <size_t... Dims>
  using value_t = property_value<work_group_size_hint_key,
                                 std::integral_constant<size_t, Dims>...>;
};

struct sub_group_size_key
    : detail::compile_time_property_key<detail::PropKind::SubGroupSize> {
  template <uint32_t Size>
  using value_t = property_value<sub_group_size_key,
                                 std::integral_constant<uint32_t, Size>>;
};

struct device_has_key
    : detail::compile_time_property_key<detail::PropKind::DeviceHas> {
  template <aspect... Aspects>
  using value_t = property_value<device_has_key,
                                 std::integral_constant<aspect, Aspects>...>;
};

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

struct max_work_group_size_key
    : detail::compile_time_property_key<detail::PropKind::MaxWorkGroupSize> {
  template <size_t... Dims>
  using value_t = property_value<max_work_group_size_key,
                                 std::integral_constant<size_t, Dims>...>;
};

struct max_linear_work_group_size_key
    : detail::compile_time_property_key<
          detail::PropKind::MaxLinearWorkGroupSize> {
  template <size_t Size>
  using value_t = property_value<max_linear_work_group_size_key,
                                 std::integral_constant<size_t, Size>>;
};

template <size_t Dim0, size_t... Dims>
struct property_value<work_group_size_key, std::integral_constant<size_t, Dim0>,
                      std::integral_constant<size_t, Dims>...>
    : detail::property_base<
          property_value<work_group_size_key,
                         std::integral_constant<size_t, Dim0>,
                         std::integral_constant<size_t, Dims>...>,
          detail::PropKind::WorkGroupSize, work_group_size_key> {
  static_assert(
      sizeof...(Dims) + 1 <= 3,
      "work_group_size property currently only supports up to three values.");
  static_assert(detail::AllNonZero<Dim0, Dims...>::value,
                "work_group_size property must only contain non-zero values.");

  constexpr size_t operator[](int Dim) const {
    return std::array<size_t, sizeof...(Dims) + 1>{Dim0, Dims...}[Dim];
  }
};

template <size_t Dim0, size_t... Dims>
struct property_value<work_group_size_hint_key,
                      std::integral_constant<size_t, Dim0>,
                      std::integral_constant<size_t, Dims>...>
    : detail::property_base<
          property_value<work_group_size_hint_key,
                         std::integral_constant<size_t, Dim0>,
                         std::integral_constant<size_t, Dims>...>,
          detail::PropKind::WorkGroupSizeHint, work_group_size_hint_key> {
  static_assert(sizeof...(Dims) + 1 <= 3,
                "work_group_size_hint property currently "
                "only supports up to three values.");
  static_assert(
      detail::AllNonZero<Dim0, Dims...>::value,
      "work_group_size_hint property must only contain non-zero values.");

  constexpr size_t operator[](int Dim) const {
    return std::array<size_t, sizeof...(Dims) + 1>{Dim0, Dims...}[Dim];
  }
};

template <uint32_t Size>
struct property_value<sub_group_size_key,
                      std::integral_constant<uint32_t, Size>>
    : detail::property_base<
          property_value<sub_group_size_key,
                         std::integral_constant<uint32_t, Size>>,
          detail::PropKind::SubGroupSize, sub_group_size_key> {
  static_assert(Size != 0,
                "sub_group_size_key property must contain a non-zero value.");

  using value_t = std::integral_constant<uint32_t, Size>;
  static constexpr uint32_t value = Size;
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

template <int Dims>
struct property_value<nd_range_kernel_key, std::integral_constant<int, Dims>>
    : detail::property_base<property_value<nd_range_kernel_key,
                                           std::integral_constant<int, Dims>>,
                            detail::PropKind::NDRangeKernel,
                            nd_range_kernel_key> {
  static_assert(
      Dims >= 1 && Dims <= 3,
      "nd_range_kernel_key property must use dimension of 1, 2 or 3.");

  using value_t = int;
  static constexpr int dimensions = Dims;
};

template <>
struct property_value<single_task_kernel_key>
    : detail::property_base<property_value<single_task_kernel_key>,
                            detail::PropKind::SingleTaskKernel,
                            single_task_kernel_key> {};

template <size_t Dim0, size_t... Dims>
struct property_value<max_work_group_size_key,
                      std::integral_constant<size_t, Dim0>,
                      std::integral_constant<size_t, Dims>...>
    : detail::property_base<
          property_value<max_work_group_size_key,
                         std::integral_constant<size_t, Dim0>,
                         std::integral_constant<size_t, Dims>...>,
          detail::PropKind::MaxWorkGroupSize, max_work_group_size_key> {
  static_assert(sizeof...(Dims) + 1 <= 3,
                "max_work_group_size property currently "
                "only supports up to three values.");
  static_assert(
      detail::AllNonZero<Dim0, Dims...>::value,
      "max_work_group_size property must only contain non-zero values.");

  constexpr size_t operator[](int Dim) const {
    return std::array<size_t, sizeof...(Dims) + 1>{Dim0, Dims...}[Dim];
  }
};

template <>
struct property_value<max_linear_work_group_size_key>
    : detail::property_base<property_value<max_linear_work_group_size_key>,
                            detail::PropKind::MaxLinearWorkGroupSize,
                            max_linear_work_group_size_key> {};

template <size_t Dim0, size_t... Dims>
inline constexpr work_group_size_key::value_t<Dim0, Dims...> work_group_size;

template <size_t Dim0, size_t... Dims>
inline constexpr work_group_size_hint_key::value_t<Dim0, Dims...>
    work_group_size_hint;

template <uint32_t Size>
inline constexpr sub_group_size_key::value_t<Size> sub_group_size;

template <aspect... Aspects>
inline constexpr device_has_key::value_t<Aspects...> device_has;

template <int Dims>
inline constexpr nd_range_kernel_key::value_t<Dims> nd_range_kernel;

inline constexpr single_task_kernel_key::value_t single_task_kernel;

template <size_t Dim0, size_t... Dims>
inline constexpr max_work_group_size_key::value_t<Dim0, Dims...>
    max_work_group_size;

template <size_t Size>
inline constexpr max_linear_work_group_size_key::value_t<Size>
    max_linear_work_group_size;

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

template <> struct is_property_key<work_group_progress_key> : std::true_type {};
template <> struct is_property_key<sub_group_progress_key> : std::true_type {};
template <> struct is_property_key<work_item_progress_key> : std::true_type {};

namespace detail {

template <size_t... Dims>
struct HasCompileTimeEffect<work_group_size_key::value_t<Dims...>>
    : std::true_type {};
template <size_t... Dims>
struct HasCompileTimeEffect<work_group_size_hint_key::value_t<Dims...>>
    : std::true_type {};
template <uint32_t Size>
struct HasCompileTimeEffect<sub_group_size_key::value_t<Size>>
    : std::true_type {};
template <sycl::aspect... Aspects>
struct HasCompileTimeEffect<device_has_key::value_t<Aspects...>>
    : std::true_type {};

template <size_t Dim0, size_t... Dims>
struct PropertyMetaInfo<work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-work-group-size";
  static constexpr const char *value = SizeListToStr<Dim0, Dims...>::value;
};
template <size_t Dim0, size_t... Dims>
struct PropertyMetaInfo<work_group_size_hint_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-work-group-size-hint";
  static constexpr const char *value = SizeListToStr<Dim0, Dims...>::value;
};
template <uint32_t Size>
struct PropertyMetaInfo<sub_group_size_key::value_t<Size>> {
  static constexpr const char *name = "sycl-sub-group-size";
  static constexpr uint32_t value = Size;
};
template <aspect... Aspects>
struct PropertyMetaInfo<device_has_key::value_t<Aspects...>> {
  static constexpr const char *name = "sycl-device-has";
  static constexpr const char *value =
      SizeListToStr<static_cast<size_t>(Aspects)...>::value;
};
template <int Dims>
struct PropertyMetaInfo<nd_range_kernel_key::value_t<Dims>> {
  static constexpr const char *name = "sycl-nd-range-kernel";
  static constexpr int value = Dims;
};
template <> struct PropertyMetaInfo<single_task_kernel_key::value_t> {
  static constexpr const char *name = "sycl-single-task-kernel";
  static constexpr int value = 0;
};
template <size_t Dim0, size_t... Dims>
struct PropertyMetaInfo<max_work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-max-work-group-size";
  static constexpr const char *value = SizeListToStr<Dim0, Dims...>::value;
};
template <size_t Size>
struct PropertyMetaInfo<max_linear_work_group_size_key::value_t<Size>> {
  static constexpr const char *name = "sycl-max-linear-work-group-size";
  static constexpr size_t value = Size;
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

// Trait for property compile-time meta names and values.
template <typename PropertyT> struct WGSizePropertyMetaInfo {
  static constexpr std::array<size_t, 0> WGSize = {};
  static constexpr size_t LinearSize = 0;
};

template <size_t Dim0, size_t... Dims>
struct WGSizePropertyMetaInfo<work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr std::array<size_t, sizeof...(Dims) + 1> WGSize = {Dim0,
                                                                     Dims...};
  static constexpr size_t LinearSize = (Dim0 * ... * Dims);
};

template <size_t Dim0, size_t... Dims>
struct WGSizePropertyMetaInfo<max_work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr std::array<size_t, sizeof...(Dims) + 1> WGSize = {Dim0,
                                                                     Dims...};
  static constexpr size_t LinearSize = (Dim0 * ... * Dims);
};

// Get the value of a work-group size related property from a property list
template <typename PropKey, typename PropertiesT>
struct GetWGPropertyFromPropList {};

template <typename PropKey, typename... PropertiesT>
struct GetWGPropertyFromPropList<PropKey, std::tuple<PropertiesT...>> {
  using prop_val_t = std::conditional_t<
      ContainsProperty<PropKey, std::tuple<PropertiesT...>>::value,
      typename FindCompileTimePropertyValueType<
          PropKey, std::tuple<PropertiesT...>>::type,
      void>;
  static constexpr auto WGSize =
      WGSizePropertyMetaInfo<std::remove_const_t<prop_val_t>>::WGSize;
  static constexpr size_t LinearSize =
      WGSizePropertyMetaInfo<std::remove_const_t<prop_val_t>>::LinearSize;
};

// If work_group_size and max_work_group_size coexist, check that the
// dimensionality matches and that the required work-group size doesn't
// trivially exceed the maximum size.
template <typename Properties>
struct ConflictingProperties<max_work_group_size_key, Properties>
    : std::false_type {
  using WGSizeVal = GetWGPropertyFromPropList<work_group_size_key, Properties>;
  using MaxWGSizeVal =
      GetWGPropertyFromPropList<max_work_group_size_key, Properties>;
  // If work_group_size_key doesn't exist in the list of properties, WGSize is
  // an empty array and so Dims == 0.
  static constexpr size_t Dims = WGSizeVal::WGSize.size();
  static_assert(
      Dims == 0 || Dims == MaxWGSizeVal::WGSize.size(),
      "work_group_size and max_work_group_size dimensionality must match");
  static_assert(Dims < 1 || WGSizeVal::WGSize[0] <= MaxWGSizeVal::WGSize[0],
                "work_group_size must not exceed max_work_group_size");
  static_assert(Dims < 2 || WGSizeVal::WGSize[1] <= MaxWGSizeVal::WGSize[1],
                "work_group_size must not exceed max_work_group_size");
  static_assert(Dims < 3 || WGSizeVal::WGSize[2] <= MaxWGSizeVal::WGSize[2],
                "work_group_size must not exceed max_work_group_size");
};

// If work_group_size and max_linear_work_group_size coexist, check that the
// required linear work-group size doesn't trivially exceed the maximum size.
template <typename Properties>
struct ConflictingProperties<max_linear_work_group_size_key, Properties>
    : std::false_type {
  using WGSizeVal = GetWGPropertyFromPropList<work_group_size_key, Properties>;
  using MaxLinearWGSizeVal =
      GetPropertyValueFromPropList<max_linear_work_group_size_key, size_t, void,
                                   Properties>;
  static_assert(WGSizeVal::WGSize.empty() ||
                    WGSizeVal::LinearSize <= MaxLinearWGSizeVal::value,
                "work_group_size must not exceed max_linear_work_group_size");
};

} // namespace detail
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(PROP)                                \
  [[__sycl_detail__::add_ir_attributes_function(                               \
      sycl::ext::oneapi::experimental::detail::PropertyMetaInfo<               \
          std::remove_cv_t<std::remove_reference_t<decltype(PROP)>>>::name,    \
      sycl::ext::oneapi::experimental::detail::PropertyMetaInfo<               \
          std::remove_cv_t<std::remove_reference_t<decltype(PROP)>>>::value)]]
#else
#define SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(PROP)
#endif
