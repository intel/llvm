//==---------------- types.hpp --- SYCL types ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>              // for decorated, address_space
#include <sycl/aliases.hpp>                    // for half, cl_char, cl_int
#include <sycl/detail/common.hpp>              // for ArrayCreator, RepeatV...
#include <sycl/detail/defines_elementary.hpp>  // for __SYCL2020_DEPRECATED
#include <sycl/detail/generic_type_lists.hpp>  // for vector_basic_list
#include <sycl/detail/generic_type_traits.hpp> // for is_sigeninteger, is_s...
#include <sycl/detail/is_device_copyable.hpp>
#include <sycl/detail/type_list.hpp>           // for is_contained
#include <sycl/detail/type_traits.hpp>         // for is_floating_point
#include <sycl/exception.hpp>                  // for make_error_code, errc
#include <sycl/half_type.hpp>                  // for StorageT, half, Vec16...
#include <sycl/marray.hpp>                     // for __SYCL_BINOP, __SYCL_...
#include <sycl/multi_ptr.hpp>                  // for multi_ptr
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/vector_preview.hpp> // for sycl::vec and swizzles
#else
#include <sycl/vector.hpp> // for sycl::vec and swizzles
#endif

#include <sycl/ext/oneapi/bfloat16.hpp> // bfloat16

namespace sycl {
inline namespace _V1 {
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
