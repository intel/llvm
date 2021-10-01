//==----------- functional.hpp --- SYCL functional -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/functional.hpp>

#include <functional>

__SYCL_OPEN_NS() {
namespace ext {
namespace oneapi {

template <typename T = void> using plus = std::plus<T>;
template <typename T = void> using multiplies = std::multiplies<T>;
template <typename T = void> using bit_or = std::bit_or<T>;
template <typename T = void> using bit_xor = std::bit_xor<T>;
template <typename T = void> using bit_and = std::bit_and<T>;
template <typename T = void> using maximum = __sycl_ns::maximum<T>;
template <typename T = void> using minimum = __sycl_ns::minimum<T>;

} // namespace oneapi
} // namespace ext

#ifdef __SYCL_DEVICE_ONLY__
namespace detail {

struct GroupOpISigned {};
struct GroupOpIUnsigned {};
struct GroupOpFP {};

template <typename T, typename = void> struct GroupOpTag;

template <typename T>
struct GroupOpTag<T, detail::enable_if_t<detail::is_sigeninteger<T>::value>> {
  using type = GroupOpISigned;
};

template <typename T>
struct GroupOpTag<T, detail::enable_if_t<detail::is_sugeninteger<T>::value>> {
  using type = GroupOpIUnsigned;
};

template <typename T>
struct GroupOpTag<T, detail::enable_if_t<detail::is_sgenfloat<T>::value>> {
  using type = GroupOpFP;
};

#define __SYCL_CALC_OVERLOAD(GroupTag, SPIRVOperation, BinaryOperation)        \
  template <typename T, __spv::GroupOperation O, __spv::Scope::Flag S>         \
  static T calc(GroupTag, T x, BinaryOperation) {                              \
    using ConvertedT = detail::ConvertToOpenCLType_t<T>;                       \
                                                                               \
    using OCLT =                                                               \
        conditional_t<std::is_same<ConvertedT, cl_char>() ||                   \
                          std::is_same<ConvertedT, cl_short>(),                \
                      cl_int,                                                  \
                      conditional_t<std::is_same<ConvertedT, cl_uchar>() ||    \
                                        std::is_same<ConvertedT, cl_ushort>(), \
                                    cl_uint, ConvertedT>>;                     \
    OCLT Arg = x;                                                              \
    OCLT Ret =                                                                 \
        __spirv_Group##SPIRVOperation(S, static_cast<unsigned int>(O), Arg);   \
    return Ret;                                                                \
  }

// calc for sycl function objects
__SYCL_CALC_OVERLOAD(GroupOpISigned, SMin, __sycl_ns::minimum<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, UMin, __sycl_ns::minimum<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FMin, __sycl_ns::minimum<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, SMax, __sycl_ns::maximum<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, UMax, __sycl_ns::maximum<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FMax, __sycl_ns::maximum<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, IAdd, __sycl_ns::plus<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, IAdd, __sycl_ns::plus<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FAdd, __sycl_ns::plus<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, NonUniformIMul, __sycl_ns::multiplies<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, NonUniformIMul, __sycl_ns::multiplies<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, NonUniformFMul, __sycl_ns::multiplies<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, NonUniformBitwiseOr, __sycl_ns::bit_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, NonUniformBitwiseOr, __sycl_ns::bit_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, NonUniformBitwiseXor, __sycl_ns::bit_xor<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, NonUniformBitwiseXor, __sycl_ns::bit_xor<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, NonUniformBitwiseAnd, __sycl_ns::bit_and<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, NonUniformBitwiseAnd, __sycl_ns::bit_and<T>)

#undef __SYCL_CALC_OVERLOAD

template <typename T, __spv::GroupOperation O, __spv::Scope::Flag S,
          template <typename> class BinaryOperation>
static T calc(typename GroupOpTag<T>::type, T x, BinaryOperation<void>) {
  return calc<T, O, S>(typename GroupOpTag<T>::type(), x, BinaryOperation<T>());
}

} // namespace detail
#endif // __SYCL_DEVICE_ONLY__

} // __SYCL_OPEN_NS()
__SYCL_CLOSE_NS()
