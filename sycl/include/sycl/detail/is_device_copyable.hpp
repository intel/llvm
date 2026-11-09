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
template <typename... T> struct tuple;

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

template <typename... Ts>
struct is_device_copyable<sycl::detail::tuple<Ts...>>
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

/// The predicate that all the device copyability requirements enforced by the
/// implementation are written in terms of. It is is_device_copyable, except
/// that SYCL_DISABLE_DEVICE_COPYABLE_CHECKS makes it true for every type. See
/// sycl/doc/PreprocessorMacros.md.
///
/// The checks consult this instead of is_device_copyable directly so that the
/// macro does not change the value the user facing trait reports: code outside
/// of the implementation specializes and queries is_device_copyable, and it
/// must keep seeing the answer the SYCL specification gives.
#ifdef SYCL_DISABLE_DEVICE_COPYABLE_CHECKS
template <typename T> struct check_if_device_copyable : std::true_type {};
#else
template <typename T>
struct check_if_device_copyable : is_device_copyable<T> {};
#endif

template <typename T>
inline constexpr bool check_if_device_copyable_v =
    check_if_device_copyable<T>::value;

// Deprecated sycl 1.2.1 rules for now
template <typename T>
inline constexpr bool deprecated_sycl121_device_copyable_v =
    std::is_trivially_copy_constructible_v<T> &&
    std::is_trivially_destructible_v<T>;

template <typename T> struct is_legal_kernel_parameter;

template <typename T>
inline constexpr bool is_legal_kernel_parameter_v =
    is_legal_kernel_parameter<T>::value;

template <typename T> struct is_sycl_accessor : std::false_type {};

template <typename T>
inline constexpr bool is_sycl_accessor_v = is_sycl_accessor<T>::value;

#ifdef __SYCL_DEVICE_ONLY__
template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernel;
template <typename TransformedArgType, int Dims, typename KernelType>
class RoundedRangeKernelWithKH;
#endif

namespace legal_kernel_parameters {
// Some types are considered special SYCL types, such as SYCL accessors.
// These types are accepted legal parameters even if they do not satisfy the
// usual device copyability requirements.
template <typename T>
struct is_special_sycl_type
    : std::bool_constant<is_sycl_accessor_v<std::remove_cv_t<T>>> {};

template <typename T>
inline constexpr bool is_special_sycl_type_v = is_special_sycl_type<T>::value;

#ifdef __SYCL_DEVICE_ONLY__
// We need to be able to check whether a record contains special SYCL types.
template <typename T, typename = void>
struct contains_special_sycl_types : is_special_sycl_type<T> {};

template <typename T>
inline constexpr bool contains_special_sycl_types_v =
    contains_special_sycl_types<T>::value;

template <typename T, typename> struct special_sycl_types_fields;
template <typename T, unsigned... FieldIds>
struct special_sycl_types_fields<T, std::index_sequence<FieldIds...>>
    : std::bool_constant<(contains_special_sycl_types_v<
                              decltype(__builtin_field_type(T, FieldIds))> ||
                          ...)> {};

template <typename... T>
inline constexpr bool special_sycl_types_fields_v =
    special_sycl_types_fields<T...>::value;

template <typename T, typename> struct special_sycl_types_bases;
template <typename T, unsigned... BaseIds>
struct special_sycl_types_bases<T, std::index_sequence<BaseIds...>>
    : std::bool_constant<(contains_special_sycl_types_v<
                              decltype(__builtin_base_type(T, BaseIds))> ||
                          ...)> {};

template <typename... T>
inline constexpr bool special_sycl_types_bases_v =
    special_sycl_types_bases<T...>::value;

template <typename T>
struct contains_special_sycl_types<
    T, std::void_t<decltype(__builtin_num_fields(T) + __builtin_num_bases(T))>>
    : std::bool_constant<
          is_special_sycl_type_v<T> ||
          special_sycl_types_fields_v<
              T, std::make_index_sequence<__builtin_num_fields(T)>> ||
          special_sycl_types_bases_v<
              T, std::make_index_sequence<__builtin_num_bases(T)>>> {};

template <typename T, std::size_t N>
struct contains_special_sycl_types<T[N]> : contains_special_sycl_types<T> {};

// In the case where any fields or base is a special SYCL type, we need to check
// whether ALL fields and bases are legal kernel parameters.
// This catches cases where a record contains special SYCL types but not all of
// its fields and bases are legal kernel parameters, but does not catch cases
// where a record containing special SYCL types also contains illegal member
// functions.
template <typename T, typename = void>
struct check_fields_and_bases : std::false_type {};

template <typename T>
inline constexpr bool check_fields_and_bases_v =
    check_fields_and_bases<T>::value;

template <typename T, typename> struct check_fields;
template <typename T, unsigned... FieldIds>
struct check_fields<T, std::index_sequence<FieldIds...>>
    : std::bool_constant<((is_legal_kernel_parameter_v<std::remove_cv_t<
                               decltype(__builtin_field_type(T, FieldIds))>>) &&
                          ...)> {};

template <typename... T>
inline constexpr bool check_fields_v = check_fields<T...>::value;

template <typename T, typename> struct check_bases;
template <typename T, unsigned... BaseIds>
struct check_bases<T, std::index_sequence<BaseIds...>>
    : std::bool_constant<((is_legal_kernel_parameter_v<std::remove_cv_t<
                               decltype(__builtin_base_type(T, BaseIds))>>) &&
                          ...)> {};

template <typename... T>
inline constexpr bool check_bases_v = check_bases<T...>::value;

template <typename T>
struct check_fields_and_bases<
    T, std::void_t<decltype(__builtin_num_fields(T) + __builtin_num_bases(T))>>
    : std::bool_constant<
          check_fields_v<T,
                         std::make_index_sequence<__builtin_num_fields(T)>> &&
          check_bases_v<T, std::make_index_sequence<__builtin_num_bases(T)>>> {
};

template <typename T, std::size_t N>
struct check_fields_and_bases<T[N]> : check_fields_and_bases<T> {};

template <typename T, typename> struct validate_fields;
template <typename T, unsigned... FieldIds>
struct validate_fields<T, std::index_sequence<FieldIds...>> {
  static_assert(
      ((is_legal_kernel_parameter_v<
            std::remove_cv_t<decltype(__builtin_field_type(T, FieldIds))>> &&
        ...)),
      "The specified type is not a legal kernel parameter");
};

// Insert static assertions to validate fields and bases of a type at compile
// time. Used by kernel_launch_helper.hpp when wrapping kernel functors
template <typename T, typename> struct validate_bases;
template <typename T, unsigned... BaseIds>
struct validate_bases<T, std::index_sequence<BaseIds...>> {
  static_assert(
      ((is_legal_kernel_parameter_v<
            std::remove_cv_t<decltype(__builtin_base_type(T, BaseIds))>> &&
        ...)),
      "The specified type is not a legal kernel parameter");
};

// All the captures of a lambda or functor of type FuncT passed to a kernel
// must be legal kernel parameters.
template <typename FuncT>
struct validate_functor
    : validate_fields<FuncT,
                      std::make_index_sequence<__builtin_num_fields(FuncT)>>,
      validate_bases<FuncT,
                     std::make_index_sequence<__builtin_num_bases(FuncT)>> {};

// Below are two specializations for validate_functor when a kernel
// lambda is wrapped after range rounding optimization.
template <typename TransformedArgType, int Dims, typename KernelType>
struct validate_functor<
    RoundedRangeKernel<TransformedArgType, Dims, KernelType>>
    : validate_functor<KernelType> {};

template <typename TransformedArgType, int Dims, typename KernelType>
struct validate_functor<
    RoundedRangeKernelWithKH<TransformedArgType, Dims, KernelType>>
    : validate_functor<KernelType> {};

#endif
} // namespace legal_kernel_parameters

// Is legal kernel parameter as defined by the SYCL-2020 specification.
// (Copied here)
//  - Any device copyable type is a legal parameter type.
//  - The following SYCL types are legal parameter types:
//    - accessor when templated with target::device;
//    - accessor when templated with any of the deprecated parameters:
//      target::global_buffer, target::constant_buffer, or target::local;
//    - local_accessor;
//    - unsampled_image_accessor when templated with image_target::device;
//    - sampled_image_accessor when templated with image_target::device;
//    - stream;
//    - id;
//    - range;
//    - marray<T, NumElements> when T is device copyable;
//    - vec<T, NumElements>.
// - An array of element types T is a legal parameter type if T is a legal
//   parameter type.
// - A class type S with a non-static member variable of type T is a legal
//   parameter type if T is a legal parameter type and if S would otherwise be
//   a legal parameter type aside from this member variable.
// - A class type S with a non-virtual base class of type T is a legal
//   parameter type if T is a legal parameter type and if S would otherwise be
//   a legal parameter type aside from this base class.
template <typename T>
struct is_legal_kernel_parameter
#ifdef __SYCL_DEVICE_ONLY__
    : std::bool_constant<
          check_if_device_copyable_v<T> ||
          legal_kernel_parameters::is_special_sycl_type_v<T> ||
          (legal_kernel_parameters::contains_special_sycl_types_v<T> &&
           legal_kernel_parameters::check_fields_and_bases_v<T>) ||
          deprecated_sycl121_device_copyable_v<T>> {
};

#else
    : std::bool_constant<check_if_device_copyable_v<T> ||
                         legal_kernel_parameters::is_special_sycl_type_v<T> ||
                         deprecated_sycl121_device_copyable_v<T>> {
};
#endif // __SYCL_DEVICE_ONLY__

template <typename T, std::size_t N>
struct is_legal_kernel_parameter<T[N]> : is_legal_kernel_parameter<T> {};
} // namespace detail
} // namespace _V1
} // namespace sycl
