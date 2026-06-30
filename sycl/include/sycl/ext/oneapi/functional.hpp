//==----------- functional.hpp --- SYCL functional -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp> // for __SYCL2020_DEPRECATED
#include <sycl/detail/spirv.hpp>
#include <sycl/functional.hpp> // for maximum, minimum

#include <functional> // for bit_and, bit_or, bit_xor, multiplies

namespace sycl {
inline namespace _V1 {

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
namespace ext::oneapi {

template <typename T = void>
using plus __SYCL2020_DEPRECATED("Use sycl::plus<> instead") = std::plus<T>;
template <typename T = void>
using multiplies __SYCL2020_DEPRECATED("Use sycl::multiplies<> instead") =
    std::multiplies<T>;
template <typename T = void>
using bit_or
    __SYCL2020_DEPRECATED("Use sycl::bit_or<> instead") = std::bit_or<T>;
template <typename T = void>
using bit_xor
    __SYCL2020_DEPRECATED("Use sycl::bit_xor<> instead") = std::bit_xor<T>;
template <typename T = void>
using bit_and
    __SYCL2020_DEPRECATED("Use sycl::bit_and<> instead") = std::bit_and<T>;
template <typename T = void>
using maximum
    __SYCL2020_DEPRECATED("Use sycl::maximum<> instead") = sycl::maximum<T>;
template <typename T = void>
using minimum
    __SYCL2020_DEPRECATED("Use sycl::minimum<> instead") = sycl::minimum<T>;
} // namespace ext::oneapi
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

// TODO: The group operation helpers below are not extension-specific; consider
// moving them to a non-extension header (e.g. sycl/functional.hpp).
#ifdef __SYCL_DEVICE_ONLY__
namespace detail {

struct GroupOpISigned {};
struct GroupOpIUnsigned {};
struct GroupOpFP {};
struct GroupOpC {};
struct GroupOpBool {};

template <typename T, typename = void> struct GroupOpTag;

template <typename T>
struct GroupOpTag<T, std::enable_if_t<detail::is_sigeninteger_v<T>>> {
  using type = GroupOpISigned;
};

template <typename T>
struct GroupOpTag<T, std::enable_if_t<detail::is_sugeninteger_v<T>>> {
  using type = GroupOpIUnsigned;
};

template <typename T>
struct GroupOpTag<T, std::enable_if_t<detail::is_sgenfloat_v<T>>> {
  using type = GroupOpFP;
};

template <typename T>
struct GroupOpTag<T, std::enable_if_t<detail::is_genbool_v<T>>> {
  using type = GroupOpBool;
};

// GroupOpC (std::complex) is handled in sycl/stl_wrappers/complex.

#define __SYCL_CALC_OVERLOAD(GroupTag, SPIRVOperation, BinaryOperation)        \
  template <__spv::GroupOperation O, typename Group, typename T>               \
  static T calc(Group g, GroupTag, T x, BinaryOperation) {                     \
    return sycl::detail::spirv::Group##SPIRVOperation<O>(g, x);                \
  }

// calc for sycl function objects
__SYCL_CALC_OVERLOAD(GroupOpISigned, SMin, sycl::minimum<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, UMin, sycl::minimum<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FMin, sycl::minimum<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, SMax, sycl::maximum<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, UMax, sycl::maximum<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FMax, sycl::maximum<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, IAdd, sycl::plus<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, IAdd, sycl::plus<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FAdd, sycl::plus<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, IMul, sycl::multiplies<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, IMul, sycl::multiplies<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FMul, sycl::multiplies<T>)
__SYCL_CALC_OVERLOAD(GroupOpC, CMulINTEL, sycl::multiplies<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, BitwiseOr, sycl::bit_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, BitwiseOr, sycl::bit_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, BitwiseXor, sycl::bit_xor<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, BitwiseXor, sycl::bit_xor<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, BitwiseAnd, sycl::bit_and<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, BitwiseAnd, sycl::bit_and<T>)

__SYCL_CALC_OVERLOAD(GroupOpBool, LogicalAnd, sycl::logical_and<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, LogicalAnd, sycl::logical_and<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, LogicalAnd, sycl::logical_and<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, LogicalAnd, sycl::logical_and<T>)

__SYCL_CALC_OVERLOAD(GroupOpBool, LogicalOr, sycl::logical_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, LogicalOr, sycl::logical_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, LogicalOr, sycl::logical_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, LogicalOr, sycl::logical_or<T>)

#undef __SYCL_CALC_OVERLOAD

template <__spv::GroupOperation O, typename Group, typename T,
          template <typename> class BinaryOperation>
static T calc(Group g, typename GroupOpTag<T>::type, T x,
              BinaryOperation<void>) {
  return calc<O>(g, typename GroupOpTag<T>::type(), x, BinaryOperation<T>());
}

} // namespace detail
#endif // __SYCL_DEVICE_ONLY__

} // namespace _V1
} // namespace sycl
