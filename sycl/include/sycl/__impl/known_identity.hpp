//==----------- known_identity.hpp -----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/detail/type_traits.hpp>
#include <functional>
#include <limits>
#include <type_traits>

namespace __sycl_internal {
inline namespace __v1 {
namespace detail {

template <typename T, class BinaryOperation>
using IsPlus =
    bool_constant<std::is_same<BinaryOperation, __sycl_internal::plus<T>>::value ||
                  std::is_same<BinaryOperation, __sycl_internal::plus<void>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::plus<T>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::plus<void>>::value>;

template <typename T, class BinaryOperation>
using IsMultiplies = bool_constant<
    std::is_same<BinaryOperation, __sycl_internal::multiplies<T>>::value ||
    std::is_same<BinaryOperation, __sycl_internal::multiplies<void>>::value ||
    std::is_same<BinaryOperation, ONEAPI::multiplies<T>>::value ||
    std::is_same<BinaryOperation, ONEAPI::multiplies<void>>::value>;

template <typename T, class BinaryOperation>
using IsMinimum =
    bool_constant<std::is_same<BinaryOperation, __sycl_internal::minimum<T>>::value ||
                  std::is_same<BinaryOperation, __sycl_internal::minimum<void>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::minimum<T>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::minimum<void>>::value>;

template <typename T, class BinaryOperation>
using IsMaximum =
    bool_constant<std::is_same<BinaryOperation, __sycl_internal::maximum<T>>::value ||
                  std::is_same<BinaryOperation, __sycl_internal::maximum<void>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::maximum<T>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::maximum<void>>::value>;

template <typename T, class BinaryOperation>
using IsBitOR =
    bool_constant<std::is_same<BinaryOperation, __sycl_internal::bit_or<T>>::value ||
                  std::is_same<BinaryOperation, __sycl_internal::bit_or<void>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::bit_or<T>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::bit_or<void>>::value>;

template <typename T, class BinaryOperation>
using IsBitXOR =
    bool_constant<std::is_same<BinaryOperation, __sycl_internal::bit_xor<T>>::value ||
                  std::is_same<BinaryOperation, __sycl_internal::bit_xor<void>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::bit_xor<T>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::bit_xor<void>>::value>;

template <typename T, class BinaryOperation>
using IsBitAND =
    bool_constant<std::is_same<BinaryOperation, __sycl_internal::bit_and<T>>::value ||
                  std::is_same<BinaryOperation, __sycl_internal::bit_and<void>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::bit_and<T>>::value ||
                  std::is_same<BinaryOperation, ONEAPI::bit_and<void>>::value>;

// Identity = 0
template <typename T, class BinaryOperation>
using IsZeroIdentityOp = bool_constant<
    (is_sgeninteger<T>::value &&
     (IsPlus<T, BinaryOperation>::value || IsBitOR<T, BinaryOperation>::value ||
      IsBitXOR<T, BinaryOperation>::value)) ||
    (is_sgenfloat<T>::value && IsPlus<T, BinaryOperation>::value)>;

// Identity = 1
template <typename T, class BinaryOperation>
using IsOneIdentityOp =
    bool_constant<(is_sgeninteger<T>::value || is_sgenfloat<T>::value) &&
                  IsMultiplies<T, BinaryOperation>::value>;

// Identity = ~0
template <typename T, class BinaryOperation>
using IsOnesIdentityOp = bool_constant<is_sgeninteger<T>::value &&
                                       IsBitAND<T, BinaryOperation>::value>;

// Identity = <max possible value>
template <typename T, class BinaryOperation>
using IsMinimumIdentityOp =
    bool_constant<(is_sgeninteger<T>::value || is_sgenfloat<T>::value) &&
                  IsMinimum<T, BinaryOperation>::value>;

// Identity = <min possible value>
template <typename T, class BinaryOperation>
using IsMaximumIdentityOp =
    bool_constant<(is_sgeninteger<T>::value || is_sgenfloat<T>::value) &&
                  IsMaximum<T, BinaryOperation>::value>;

template <typename T, class BinaryOperation>
using IsKnownIdentityOp =
    bool_constant<IsZeroIdentityOp<T, BinaryOperation>::value ||
                  IsOneIdentityOp<T, BinaryOperation>::value ||
                  IsOnesIdentityOp<T, BinaryOperation>::value ||
                  IsMinimumIdentityOp<T, BinaryOperation>::value ||
                  IsMaximumIdentityOp<T, BinaryOperation>::value>;

template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity_impl
    : std::integral_constant<
          bool, IsKnownIdentityOp<AccumulatorT, BinaryOperation>::value> {};

template <typename BinaryOperation, typename AccumulatorT, typename = void>
struct known_identity_impl {};

/// Returns zero as identity for ADD, OR, XOR operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           typename std::enable_if<IsZeroIdentityOp<
                               AccumulatorT, BinaryOperation>::value>::type> {
  static constexpr AccumulatorT value = 0;
};

template <typename BinaryOperation>
struct known_identity_impl<BinaryOperation, half,
                           typename std::enable_if<IsZeroIdentityOp<
                               half, BinaryOperation>::value>::type> {
  static constexpr half value =
#ifdef __SYCL_DEVICE_ONLY__
      0;
#else
      __sycl_internal::detail::host_half_impl::half_v2(static_cast<uint16_t>(0));
#endif
};

/// Returns one as identify for MULTIPLY operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           typename std::enable_if<IsOneIdentityOp<
                               AccumulatorT, BinaryOperation>::value>::type> {
  static constexpr AccumulatorT value = 1;
};

template <typename BinaryOperation>
struct known_identity_impl<BinaryOperation, half,
                           typename std::enable_if<IsOneIdentityOp<
                               half, BinaryOperation>::value>::type> {
  static constexpr half value =
#ifdef __SYCL_DEVICE_ONLY__
      1;
#else
      __sycl_internal::detail::host_half_impl::half_v2(static_cast<uint16_t>(0x3C00));
#endif
};

/// Returns bit image consisting of all ones as identity for AND operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           typename std::enable_if<IsOnesIdentityOp<
                               AccumulatorT, BinaryOperation>::value>::type> {
  static constexpr AccumulatorT value = ~static_cast<AccumulatorT>(0);
};

/// Returns maximal possible value as identity for MIN operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           typename std::enable_if<IsMinimumIdentityOp<
                               AccumulatorT, BinaryOperation>::value>::type> {
  static constexpr AccumulatorT value =
      std::numeric_limits<AccumulatorT>::has_infinity
          ? std::numeric_limits<AccumulatorT>::infinity()
          : (std::numeric_limits<AccumulatorT>::max)();
};

/// Returns minimal possible value as identity for MAX operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           typename std::enable_if<IsMaximumIdentityOp<
                               AccumulatorT, BinaryOperation>::value>::type> {
  static constexpr AccumulatorT value =
      std::numeric_limits<AccumulatorT>::has_infinity
          ? static_cast<AccumulatorT>(
                -std::numeric_limits<AccumulatorT>::infinity())
          : std::numeric_limits<AccumulatorT>::lowest();
};

} // namespace detail

// ---- has_known_identity
template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity : detail::has_known_identity_impl<
                                typename std::decay<BinaryOperation>::type,
                                typename std::decay<AccumulatorT>::type> {};

template <typename BinaryOperation, typename AccumulatorT>
__SYCL_INLINE_CONSTEXPR bool has_known_identity_v =
    __sycl_internal::has_known_identity<BinaryOperation, AccumulatorT>::value;

// ---- known_identity
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity
    : detail::known_identity_impl<typename std::decay<BinaryOperation>::type,
                                  typename std::decay<AccumulatorT>::type> {};

template <typename BinaryOperation, typename AccumulatorT>
__SYCL_INLINE_CONSTEXPR AccumulatorT known_identity_v =
    __sycl_internal::known_identity<BinaryOperation, AccumulatorT>::value;

} // namespace sycl
} // namespace __sycl_internal