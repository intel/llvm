//==------------ is_device_copyable.hpp - ----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/defines_elementary.hpp>

#include <array>
#include <optional>
#include <type_traits>
#include <variant>

/// This macro must be defined to 1 when SYCL implementation allows user
/// applications to explicitly declare certain class types as device copyable
/// by adding specializations of is_device_copyable type trait class.
#define SYCL_DEVICE_COPYABLE 1

namespace sycl {
inline namespace _V1 {
/// is_device_copyable is a user specializable class template to indicate
/// that a type T is device copyable, which means that SYCL implementation
/// may copy objects of the type T between host and device or between two
/// devices.
/// Specializing is_device_copyable such a way that
/// is_device_copyable_v<T> == true on a T that does not satisfy all
/// the requirements of a device copyable type is undefined behavior.
template <typename T> struct is_device_copyable;

namespace detail {
template <typename T, typename = void>
struct is_device_copyable_impl : std::is_trivially_copyable<T> {};

template <typename T>
struct is_device_copyable_impl<
    T, std::enable_if_t<!std::is_same_v<T, std::remove_cv_t<T>>>>
    // Cannot express this "recursion" (to take user's partial non-cv
    // specializations into account) without this helper struct.
    : is_device_copyable<std::remove_cv_t<T>> {};
} // namespace detail

template <typename T>
struct is_device_copyable : detail::is_device_copyable_impl<T> {};

// std::array<T, 0> is implicitly device copyable type.
template <typename T>
struct is_device_copyable<std::array<T, 0>> : std::true_type {};

// std::array<T, N> is implicitly device copyable type if T is device copyable.
template <typename T, std::size_t N>
struct is_device_copyable<std::array<T, N>> : is_device_copyable<T> {};

// std::optional<T> is implicitly device copyable type if T is device copyable.
template <typename T>
struct is_device_copyable<std::optional<T>> : is_device_copyable<T> {};

// std::pair<T1, T2> is implicitly device copyable type if T1 and T2 are device
// copyable.
template <typename T1, typename T2>
struct is_device_copyable<std::pair<T1, T2>>
    : std::bool_constant<is_device_copyable<T1>::value &&
                         is_device_copyable<T2>::value> {};

// std::tuple<Ts...> is implicitly device copyable type if each type T of Ts...
// is device copyable.
template <typename... Ts>
struct is_device_copyable<std::tuple<Ts...>>
    : std::bool_constant<(... && is_device_copyable<Ts>::value)> {};

// std::variant<Ts...> is implicitly device copyable type if each type T of
// Ts... is device copyable.
template <typename... Ts>
struct is_device_copyable<std::variant<Ts...>>
    : std::bool_constant<(... && is_device_copyable<Ts>::value)> {};

// array is device copyable if element type is device copyable.
template <typename T, std::size_t N>
struct is_device_copyable<T[N]> : is_device_copyable<T> {};

template <typename T>
inline constexpr bool is_device_copyable_v = is_device_copyable<T>::value;
namespace detail {
template <typename T, typename = void>
struct IsDeprecatedDeviceCopyable : std::false_type {};

// TODO: using C++ attribute [[deprecated]] or the macro __SYCL2020_DEPRECATED
// does not produce expected warning message for the type 'T'.
template <typename T>
struct __SYCL2020_DEPRECATED("This type isn't device copyable in SYCL 2020")
    IsDeprecatedDeviceCopyable<
        T, std::enable_if_t<std::is_trivially_copy_constructible_v<T> &&
                            std::is_trivially_destructible_v<T> &&
                            !is_device_copyable_v<T>>> : std::true_type {};

template <typename T, int N>
struct __SYCL2020_DEPRECATED("This type isn't device copyable in SYCL 2020")
    IsDeprecatedDeviceCopyable<T[N]> : IsDeprecatedDeviceCopyable<T> {};

#ifdef __SYCL_DEVICE_ONLY__
// Checks that the fields of the type T with indices 0 to (NumFieldsToCheck -
// 1) are device copyable.
template <typename T, unsigned NumFieldsToCheck>
struct CheckFieldsAreDeviceCopyable
    : CheckFieldsAreDeviceCopyable<T, NumFieldsToCheck - 1> {
  using FieldT = decltype(__builtin_field_type(T, NumFieldsToCheck - 1));
  static_assert(is_device_copyable_v<FieldT> ||
                    detail::IsDeprecatedDeviceCopyable<FieldT>::value,
                "The specified type is not device copyable");
};

template <typename T> struct CheckFieldsAreDeviceCopyable<T, 0> {};

// Checks that the base classes of the type T with indices 0 to
// (NumFieldsToCheck - 1) are device copyable.
template <typename T, unsigned NumBasesToCheck>
struct CheckBasesAreDeviceCopyable
    : CheckBasesAreDeviceCopyable<T, NumBasesToCheck - 1> {
  using BaseT = decltype(__builtin_base_type(T, NumBasesToCheck - 1));
  static_assert(is_device_copyable_v<BaseT> ||
                    detail::IsDeprecatedDeviceCopyable<BaseT>::value,
                "The specified type is not device copyable");
};

template <typename T> struct CheckBasesAreDeviceCopyable<T, 0> {};

// All the captures of a lambda or functor of type FuncT passed to a kernel
// must be is_device_copyable, which extends to bases and fields of FuncT.
// Fields are captures of lambda/functors and bases are possible base classes
// of functors also allowed by SYCL.
// The SYCL-2020 implementation must check each of the fields & bases of the
// type FuncT, only one level deep, which is enough to see if they are all
// device copyable by using the result of is_device_copyable returned for them.
// At this moment though the check also allowes using types for which
// (is_trivially_copy_constructible && is_trivially_destructible) returns true
// and (is_device_copyable) returns false. That is the deprecated behavior and
// is currently/temporarily supported only to not break older SYCL programs.
template <typename FuncT>
struct CheckDeviceCopyable
    : CheckFieldsAreDeviceCopyable<FuncT, __builtin_num_fields(FuncT)>,
      CheckBasesAreDeviceCopyable<FuncT, __builtin_num_bases(FuncT)> {};

template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernel;
template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernelWithKH;

// Below are two specializations for CheckDeviceCopyable when a kernel lambda
// is wrapped after range rounding optimization.
template <typename TransformedArgType, int Dims, typename KernelType>
struct CheckDeviceCopyable<
    RoundedRangeKernel<TransformedArgType, Dims, KernelType>>
    : CheckDeviceCopyable<KernelType> {};

template <typename TransformedArgType, int Dims, typename KernelType>
struct CheckDeviceCopyable<
    RoundedRangeKernelWithKH<TransformedArgType, Dims, KernelType>>
    : CheckDeviceCopyable<KernelType> {};

#endif // __SYCL_DEVICE_ONLY__
} // namespace detail
} // namespace _V1
} // namespace sycl
