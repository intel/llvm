//==--- function_launch_properties.hpp - SYCL function launch properties --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// This header extends the split kernel-property path with launch-tuning
// annotations. Keep these public property_value definitions local to this split
// header so the umbrella include reuses the same types and preserves
// decltype(...) identity without pulling the full kernel-properties machinery
// into standalone launch-property use.

#include <array>
#include <stddef.h>
#include <stdint.h>

#include <sycl/ext/oneapi/kernel_properties/function_properties.hpp>
#include <sycl/ext/oneapi/properties/property_value.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace detail {

template <typename PropKey, typename Properties> struct ConflictingProperties;

template <size_t... Xs> struct LaunchAllNonZero {
  static constexpr bool value = true;
};
template <size_t X, size_t... Xs> struct LaunchAllNonZero<X, Xs...> {
  static constexpr bool value = X > 0 && LaunchAllNonZero<Xs...>::value;
};

inline constexpr size_t LaunchDecimalBase = 10;

template <size_t... Sizes> struct LaunchSizeList {};
template <char... Chars> struct LaunchCharList {};

template <char... Chars> struct LaunchCharsToStr {
  static constexpr const char value[] = {Chars..., '\0'};
};

template <typename List, typename ParsedList, char... Chars>
struct LaunchSizeListToStrHelper;

template <size_t Value, size_t... Values, char... ParsedChars, char... Chars>
struct LaunchSizeListToStrHelper<LaunchSizeList<Value, Values...>,
                                 LaunchCharList<ParsedChars...>, Chars...>
    : LaunchSizeListToStrHelper<
          LaunchSizeList<Value / LaunchDecimalBase, Values...>,
          LaunchCharList<ParsedChars...>, '0' + (Value % LaunchDecimalBase),
          Chars...> {};

template <size_t... Values, char... ParsedChars, char... Chars>
struct LaunchSizeListToStrHelper<LaunchSizeList<0, Values...>,
                                 LaunchCharList<ParsedChars...>, Chars...>
    : LaunchSizeListToStrHelper<LaunchSizeList<Values...>,
                                LaunchCharList<ParsedChars..., Chars..., ','>> {
};

template <size_t... Values, char... ParsedChars>
struct LaunchSizeListToStrHelper<LaunchSizeList<0, Values...>,
                                 LaunchCharList<ParsedChars...>>
    : LaunchSizeListToStrHelper<LaunchSizeList<Values...>,
                                LaunchCharList<ParsedChars..., '0', ','>> {};

template <char... ParsedChars, char... Chars>
struct LaunchSizeListToStrHelper<LaunchSizeList<0>,
                                 LaunchCharList<ParsedChars...>, Chars...>
    : LaunchCharsToStr<ParsedChars..., Chars...> {};

template <char... ParsedChars>
struct LaunchSizeListToStrHelper<LaunchSizeList<0>,
                                 LaunchCharList<ParsedChars...>>
    : LaunchCharsToStr<ParsedChars..., '0'> {};

template <>
struct LaunchSizeListToStrHelper<LaunchSizeList<>, LaunchCharList<>>
    : LaunchCharsToStr<> {};

template <size_t... Sizes>
struct LaunchSizeListToStr
    : LaunchSizeListToStrHelper<LaunchSizeList<Sizes...>, LaunchCharList<>> {};

} // namespace detail

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
  static_assert(detail::LaunchAllNonZero<Dim0, Dims...>::value,
                "work_group_size property must only contain non-zero values.");

  constexpr size_t operator[](int Dim) const {
    return std::array<size_t, sizeof...(Dims) + 1>{Dim0, Dims...}[Dim];
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
      detail::LaunchAllNonZero<Dim0, Dims...>::value,
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
  static_assert(detail::LaunchAllNonZero<Dim0, Dims...>::value,
                "max_work_group_size must only contain non-zero values.");

  constexpr size_t operator[](int Dim) const {
    return std::array<size_t, sizeof...(Dims) + 1>{Dim0, Dims...}[Dim];
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
      LaunchSizeListToStr<Dim0, Dims...>::value;
};
template <size_t Dim0, size_t... Dims>
struct FunctionPropertyMetaInfo<work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-work-group-size";
  static constexpr const char *value =
      LaunchSizeListToStr<Dim0, Dims...>::value;
};

template <size_t Dim0, size_t... Dims>
struct PropertyMetaInfo<work_group_size_hint_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-work-group-size-hint";
  static constexpr const char *value =
      LaunchSizeListToStr<Dim0, Dims...>::value;
};
template <size_t Dim0, size_t... Dims>
struct FunctionPropertyMetaInfo<
    work_group_size_hint_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-work-group-size-hint";
  static constexpr const char *value =
      LaunchSizeListToStr<Dim0, Dims...>::value;
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
      LaunchSizeListToStr<Dim0, Dims...>::value;
};
template <size_t Dim0, size_t... Dims>
struct FunctionPropertyMetaInfo<
    max_work_group_size_key::value_t<Dim0, Dims...>> {
  static constexpr const char *name = "sycl-max-work-group-size";
  static constexpr const char *value =
      LaunchSizeListToStr<Dim0, Dims...>::value;
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

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl