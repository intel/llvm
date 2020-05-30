//==----------- functional.hpp --- SYCL functional -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <functional>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {

template <typename T = void> struct minimum {
  T operator()(const T &lhs, const T &rhs) const {
    return std::less<T>()(lhs, rhs) ? lhs : rhs;
  }
};

#if __cplusplus >= 201402L
template <> struct minimum<void> {
  struct is_transparent {};
  template <typename T, typename U>
  auto operator()(T &&lhs, U &&rhs) const ->
      typename std::common_type<T &&, U &&>::type {
    return std::less<>()(std::forward<const T>(lhs), std::forward<const U>(rhs))
               ? std::forward<T>(lhs)
               : std::forward<U>(rhs);
  }
};
#endif

template <typename T = void> struct maximum {
  T operator()(const T &lhs, const T &rhs) const {
    return std::greater<T>()(lhs, rhs) ? lhs : rhs;
  }
};

#if __cplusplus >= 201402L
template <> struct maximum<void> {
  struct is_transparent {};
  template <typename T, typename U>
  auto operator()(T &&lhs, U &&rhs) const ->
      typename std::common_type<T &&, U &&>::type {
    return std::greater<>()(std::forward<const T>(lhs),
                            std::forward<const U>(rhs))
               ? std::forward<T>(lhs)
               : std::forward<U>(rhs);
  }
};
#endif

template <typename T = void> using plus = std::plus<T>;
template <typename T = void> using bit_or = std::bit_or<T>;
template <typename T = void> using bit_xor = std::bit_xor<T>;
template <typename T = void> using bit_and = std::bit_and<T>;

} // namespace intel

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
    using OCLT = detail::ConvertToOpenCLType_t<T>;                             \
    OCLT Arg = x;                                                              \
    OCLT Ret =                                                                 \
        __spirv_Group##SPIRVOperation(S, static_cast<unsigned int>(O), Arg);   \
    return Ret;                                                                \
  }

__SYCL_CALC_OVERLOAD(GroupOpISigned, SMin, intel::minimum<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, UMin, intel::minimum<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FMin, intel::minimum<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, SMax, intel::maximum<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, UMax, intel::maximum<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FMax, intel::maximum<T>)
__SYCL_CALC_OVERLOAD(GroupOpISigned, IAdd, intel::plus<T>)
__SYCL_CALC_OVERLOAD(GroupOpIUnsigned, IAdd, intel::plus<T>)
__SYCL_CALC_OVERLOAD(GroupOpFP, FAdd, intel::plus<T>)

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
