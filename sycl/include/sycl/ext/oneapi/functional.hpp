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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {

template <typename T = void> using plus = std::plus<T>;
template <typename T = void> using multiplies = std::multiplies<T>;
template <typename T = void> using bit_or = std::bit_or<T>;
template <typename T = void> using bit_xor = std::bit_xor<T>;
template <typename T = void> using bit_and = std::bit_and<T>;
template <typename T = void> using maximum = sycl::maximum<T>;
template <typename T = void> using minimum = sycl::minimum<T>;

} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
  using namespace ext::oneapi;
}

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
__SYCL_CALC_OVERLOAD(GroupOpISigned, SMin, sycl::minimum<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, UMin, sycl::minimum<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FMin, sycl::minimum<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, SMax, sycl::maximum<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, UMax, sycl::maximum<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FMax, sycl::maximum<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, IAdd, sycl::plus<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, IAdd, sycl::plus<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FAdd, sycl::plus<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, NonUniformIMul, sycl::multiplies<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, NonUniformIMul, sycl::multiplies<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, NonUniformFMul, sycl::multiplies<T>)

__SYCL_CALC_OVERLOAD(GroupOpISigned, NonUniformBitwiseOr, sycl::bit_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, NonUniformBitwiseOr, sycl::bit_or<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, NonUniformBitwiseXor, sycl::bit_xor<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, NonUniformBitwiseXor, sycl::bit_xor<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, NonUniformBitwiseAnd, sycl::bit_and<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, NonUniformBitwiseAnd, sycl::bit_and<T>)

#undef __SYCL_CALC_OVERLOAD

template <typename T, __spv::GroupOperation O, __spv::Scope::Flag S,
          template <typename> class BinaryOperation>
static T calc(typename GroupOpTag<T>::type, T x, BinaryOperation<void>) {
  return calc<T, O, S>(typename GroupOpTag<T>::type(), x, BinaryOperation<T>());
}

} // namespace detail
#endif // __SYCL_DEVICE_ONLY__

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
