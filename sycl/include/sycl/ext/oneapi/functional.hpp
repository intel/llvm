//==----------- functional.hpp --- SYCL functional -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/spirv.hpp>
#include <sycl/functional.hpp> // for maximum, minimum

#include <functional> // for bit_and, bit_or, bit_xor, multiplies

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {

template <typename T = void> using plus = std::plus<T>;
template <typename T = void> using multiplies = std::multiplies<T>;
template <typename T = void> using bit_or = std::bit_or<T>;
template <typename T = void> using bit_xor = std::bit_xor<T>;
template <typename T = void> using bit_and = std::bit_and<T>;
template <typename T = void> using maximum = sycl::maximum<T>;
template <typename T = void> using minimum = sycl::minimum<T>;

} // namespace ext::oneapi

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

__SYCL_CALC_OVERLOAD(GroupOpISigned, IMulKHR, sycl::multiplies<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, IMulKHR, sycl::multiplies<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FMulKHR, sycl::multiplies<T>)
__SYCL_CALC_OVERLOAD(GroupOpC, CMulINTEL, sycl::multiplies<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, BitwiseOrKHR, sycl::bit_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, BitwiseOrKHR, sycl::bit_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, BitwiseXorKHR, sycl::bit_xor<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, BitwiseXorKHR, sycl::bit_xor<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, BitwiseAndKHR, sycl::bit_and<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, BitwiseAndKHR, sycl::bit_and<T>)

__SYCL_CALC_OVERLOAD(GroupOpBool, LogicalAndKHR, sycl::logical_and<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, LogicalAndKHR, sycl::logical_and<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, LogicalAndKHR, sycl::logical_and<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, LogicalAndKHR, sycl::logical_and<T>)

__SYCL_CALC_OVERLOAD(GroupOpBool, LogicalOrKHR, sycl::logical_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, LogicalOrKHR, sycl::logical_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, LogicalOrKHR, sycl::logical_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, LogicalOrKHR, sycl::logical_or<T>)

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
