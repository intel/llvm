//==--- function_properties.hpp - SYCL standalone function properties -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// This header is the lightweight split entry point for free-function kernel
// annotations and standalone compile-time kernel properties that do not need
// the umbrella property's property-list machinery. Keep these public
// property_value definitions here so standalone users and umbrella users
// observe the same decltype(...) while avoiding the heavier machinery on this
// path.

#include <stddef.h>
#include <stdint.h>
#include <sycl/ext/oneapi/properties/property_value.hpp>

#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename PropertyT> struct FunctionPropertyMetaInfo;
template <typename PropKey, typename Properties> struct ConflictingProperties;

template <size_t... Xs> struct FunctionPropertyAllNonZero {
  static constexpr bool value = true;
};
template <size_t X, size_t... Xs>
struct FunctionPropertyAllNonZero<X, Xs...> {
  static constexpr bool value = X > 0 && FunctionPropertyAllNonZero<Xs...>::value;
};

inline constexpr size_t FunctionPropertyDecimalBase = 10;

template <size_t... Sizes> struct FunctionPropertySizeList {};
template <char... Chars> struct FunctionPropertyCharList {};

template <char... Chars> struct FunctionPropertyCharsToStr {
  static constexpr const char value[] = {Chars..., '\0'};
};

template <typename List, typename ParsedList, char... Chars>
struct FunctionPropertySizeListToStrHelper;

template <size_t Value, size_t... Values, char... ParsedChars, char... Chars>
struct FunctionPropertySizeListToStrHelper<
    FunctionPropertySizeList<Value, Values...>,
    FunctionPropertyCharList<ParsedChars...>, Chars...>
    : FunctionPropertySizeListToStrHelper<
          FunctionPropertySizeList<Value / FunctionPropertyDecimalBase,
                                   Values...>,
          FunctionPropertyCharList<ParsedChars...>,
          '0' + (Value % FunctionPropertyDecimalBase), Chars...> {};

template <size_t... Values, char... ParsedChars, char... Chars>
struct FunctionPropertySizeListToStrHelper<
    FunctionPropertySizeList<0, Values...>,
    FunctionPropertyCharList<ParsedChars...>, Chars...>
    : FunctionPropertySizeListToStrHelper<
          FunctionPropertySizeList<Values...>,
          FunctionPropertyCharList<ParsedChars..., Chars..., ','>> {};

template <size_t... Values, char... ParsedChars>
struct FunctionPropertySizeListToStrHelper<FunctionPropertySizeList<0, Values...>,
                                           FunctionPropertyCharList<ParsedChars...>>
    : FunctionPropertySizeListToStrHelper<
          FunctionPropertySizeList<Values...>,
          FunctionPropertyCharList<ParsedChars..., '0', ','>> {};

template <char... ParsedChars, char... Chars>
struct FunctionPropertySizeListToStrHelper<FunctionPropertySizeList<0>,
                                           FunctionPropertyCharList<ParsedChars...>,
                                           Chars...>
    : FunctionPropertyCharsToStr<ParsedChars..., Chars...> {};

template <char... ParsedChars>
struct FunctionPropertySizeListToStrHelper<FunctionPropertySizeList<0>,
                                           FunctionPropertyCharList<ParsedChars...>>
    : FunctionPropertyCharsToStr<ParsedChars..., '0'> {};

template <>
struct FunctionPropertySizeListToStrHelper<FunctionPropertySizeList<>,
                                           FunctionPropertyCharList<>>
    : FunctionPropertyCharsToStr<> {};

template <size_t... Sizes>
struct FunctionPropertySizeListToStr
    : FunctionPropertySizeListToStrHelper<FunctionPropertySizeList<Sizes...>,
                                          FunctionPropertyCharList<>> {};

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
  static_assert(detail::FunctionPropertyAllNonZero<Dim0, Dims...>::value,
                "work_group_size property must only contain non-zero values.");

  constexpr size_t operator[](int Dim) const {
    constexpr size_t Values[] = {Dim0, Dims...};
    return Values[Dim];
  }

private:
  constexpr size_t size() const { return sizeof...(Dims) + 1; }

  template <typename, typename> friend struct detail::ConflictingProperties;
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
                "work_group_size_hint property currently only supports up to "
                "three values.");
  static_assert(
      detail::FunctionPropertyAllNonZero<Dim0, Dims...>::value,
      "work_group_size_hint property must only contain non-zero values.");

  constexpr size_t operator[](int Dim) const {
    constexpr size_t Values[] = {Dim0, Dims...};
    return Values[Dim];
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
                "sub_group_size property must contain a non-zero value.");

  using value_t = std::integral_constant<uint32_t, Size>;
  static constexpr uint32_t value = Size;
};

template <size_t Dim0, size_t... Dims>
struct property_value<max_work_group_size_key,
                      std::integral_constant<size_t, Dim0>,
                      std::integral_constant<size_t, Dims>...>
    : detail::property_base<
          property_value<max_work_group_size_key,
                         std::integral_constant<size_t, Dim0>,
                         std::integral_constant<size_t, Dims>...>,
          detail::PropKind::MaxWorkGroupSize, max_work_group_size_key> {
  static_assert(
      sizeof...(Dims) + 1 <= 3,
      "max_work_group_size currently only supports up to three values.");
  static_assert(detail::FunctionPropertyAllNonZero<Dim0, Dims...>::value,
                "max_work_group_size must only contain non-zero values.");

  constexpr size_t operator[](int Dim) const {
    constexpr size_t Values[] = {Dim0, Dims...};
    return Values[Dim];
  }

private:
  constexpr size_t size() const { return sizeof...(Dims) + 1; }

  template <typename, typename> friend struct detail::ConflictingProperties;
};

template <size_t Size>
struct property_value<max_linear_work_group_size_key,
                      std::integral_constant<size_t, Size>>
    : detail::property_base<
          property_value<max_linear_work_group_size_key,
                         std::integral_constant<size_t, Size>>,
          detail::PropKind::MaxLinearWorkGroupSize,
          max_linear_work_group_size_key> {
  static_assert(Size != 0,
                "max_linear_work_group_size must contain a non-zero value.");

  using value_t = std::integral_constant<size_t, Size>;
  static constexpr size_t value = Size;
};

template <size_t Dim0, size_t... Dims>
inline constexpr work_group_size_key::value_t<Dim0, Dims...> work_group_size;

template <size_t Dim0, size_t... Dims>
inline constexpr work_group_size_hint_key::value_t<Dim0, Dims...>
    work_group_size_hint;

template <uint32_t Size>
inline constexpr sub_group_size_key::value_t<Size> sub_group_size;

template <size_t Dim0, size_t... Dims>
inline constexpr max_work_group_size_key::value_t<Dim0, Dims...>
    max_work_group_size;

template <size_t Size>
inline constexpr max_linear_work_group_size_key::value_t<Size>
    max_linear_work_group_size;

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

template <size_t... Dims>
struct HasCompileTimeEffect<work_group_size_key::value_t<Dims...>>
    : std::true_type {};
template <size_t... Dims>
struct HasCompileTimeEffect<work_group_size_hint_key::value_t<Dims...>>
    : std::true_type {};
template <uint32_t Size>
struct HasCompileTimeEffect<sub_group_size_key::value_t<Size>>
    : std::true_type {};
template <size_t... Dims>
struct HasCompileTimeEffect<max_work_group_size_key::value_t<Dims...>>
    : std::true_type {};
template <size_t Size>
struct HasCompileTimeEffect<max_linear_work_group_size_key::value_t<Size>>
    : std::true_type {};

template <size_t Dim0, size_t... Dims>
struct PropertyMetaInfo<work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-work-group-size";
  static constexpr const char *value =
      FunctionPropertySizeListToStr<Dim0, Dims...>::value;
};
template <size_t Dim0, size_t... Dims>
struct FunctionPropertyMetaInfo<work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-work-group-size";
  static constexpr const char *value =
      FunctionPropertySizeListToStr<Dim0, Dims...>::value;
};

template <size_t Dim0, size_t... Dims>
struct PropertyMetaInfo<work_group_size_hint_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-work-group-size-hint";
  static constexpr const char *value =
      FunctionPropertySizeListToStr<Dim0, Dims...>::value;
};
template <size_t Dim0, size_t... Dims>
struct FunctionPropertyMetaInfo<
    work_group_size_hint_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-work-group-size-hint";
  static constexpr const char *value =
      FunctionPropertySizeListToStr<Dim0, Dims...>::value;
};

template <uint32_t Size>
struct PropertyMetaInfo<sub_group_size_key::value_t<Size>> {
  static constexpr const char *name = "sycl-sub-group-size";
  static constexpr uint32_t value = Size;
};
template <uint32_t Size>
struct FunctionPropertyMetaInfo<sub_group_size_key::value_t<Size>> {
  static constexpr const char *name = "sycl-sub-group-size";
  static constexpr uint32_t value = Size;
};

template <size_t Dim0, size_t... Dims>
struct PropertyMetaInfo<max_work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-max-work-group-size";
  static constexpr const char *value =
      FunctionPropertySizeListToStr<Dim0, Dims...>::value;
};
template <size_t Dim0, size_t... Dims>
struct FunctionPropertyMetaInfo<
    max_work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-max-work-group-size";
  static constexpr const char *value =
      FunctionPropertySizeListToStr<Dim0, Dims...>::value;
};

template <size_t Size>
struct PropertyMetaInfo<max_linear_work_group_size_key::value_t<Size>> {
  static constexpr const char *name = "sycl-max-linear-work-group-size";
  static constexpr size_t value = Size;
};
template <size_t Size>
struct FunctionPropertyMetaInfo<max_linear_work_group_size_key::value_t<Size>> {
  static constexpr const char *name = "sycl-max-linear-work-group-size";
  static constexpr size_t value = Size;
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
