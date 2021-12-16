//==----------- known_identity.hpp -----------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/functional.hpp>
#include <functional>
#include <limits>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

template <typename T, class BinaryOperation>
using IsPlus =
    bool_constant<std::is_same<BinaryOperation, sycl::plus<T>>::value ||
                  std::is_same<BinaryOperation, sycl::plus<void>>::value>;

template <typename T, class BinaryOperation>
using IsMultiplies =
    bool_constant<std::is_same<BinaryOperation, sycl::multiplies<T>>::value ||
                  std::is_same<BinaryOperation, sycl::multiplies<void>>::value>;

template <typename T, class BinaryOperation>
using IsMinimum =
    bool_constant<std::is_same<BinaryOperation, sycl::minimum<T>>::value ||
                  std::is_same<BinaryOperation, sycl::minimum<void>>::value>;

template <typename T, class BinaryOperation>
using IsMaximum =
    bool_constant<std::is_same<BinaryOperation, sycl::maximum<T>>::value ||
                  std::is_same<BinaryOperation, sycl::maximum<void>>::value>;

template <typename T, class BinaryOperation>
using IsBitAND =
    bool_constant<std::is_same<BinaryOperation, sycl::bit_and<T>>::value ||
                  std::is_same<BinaryOperation, sycl::bit_and<void>>::value>;

template <typename T, class BinaryOperation>
using IsBitOR =
    bool_constant<std::is_same<BinaryOperation, sycl::bit_or<T>>::value ||
                  std::is_same<BinaryOperation, sycl::bit_or<void>>::value>;

template <typename T, class BinaryOperation>
using IsBitXOR =
    bool_constant<std::is_same<BinaryOperation, sycl::bit_xor<T>>::value ||
                  std::is_same<BinaryOperation, sycl::bit_xor<void>>::value>;

template <typename T, class BinaryOperation>
using IsLogicalAND = bool_constant<
    std::is_same<BinaryOperation, sycl::logical_and<T>>::value ||
    std::is_same<BinaryOperation, sycl::logical_and<void>>::value>;

template <typename T, class BinaryOperation>
using IsLogicalOR =
    bool_constant<std::is_same<BinaryOperation, sycl::logical_or<T>>::value ||
                  std::is_same<BinaryOperation, sycl::logical_or<void>>::value>;

// Identity = 0
template <typename T, class BinaryOperation>
using IsZeroIdentityOp =
    bool_constant<(is_geninteger<T>::value &&
                   (IsPlus<T, BinaryOperation>::value ||
                    IsBitOR<T, BinaryOperation>::value ||
                    IsBitXOR<T, BinaryOperation>::value)) ||
                  (is_genfloat<T>::value && IsPlus<T, BinaryOperation>::value)>;

// Identity = 1
template <typename T, class BinaryOperation>
using IsOneIdentityOp =
    bool_constant<(is_geninteger<T>::value || is_genfloat<T>::value) &&
                  IsMultiplies<T, BinaryOperation>::value>;

// Identity = ~0
template <typename T, class BinaryOperation>
using IsOnesIdentityOp = bool_constant<is_geninteger<T>::value &&
                                       IsBitAND<T, BinaryOperation>::value>;

// Identity = <max possible value>
template <typename T, class BinaryOperation>
using IsMinimumIdentityOp =
    bool_constant<(is_geninteger<T>::value || is_genfloat<T>::value) &&
                  IsMinimum<T, BinaryOperation>::value>;

// Identity = <min possible value>
template <typename T, class BinaryOperation>
using IsMaximumIdentityOp =
    bool_constant<(is_geninteger<T>::value || is_genfloat<T>::value) &&
                  IsMaximum<T, BinaryOperation>::value>;

// Identity = false
template <typename T, class BinaryOperation>
using IsFalseIdentityOp = bool_constant<IsLogicalOR<T, BinaryOperation>::value>;

// Identity = true
template <typename T, class BinaryOperation>
using IsTrueIdentityOp = bool_constant<IsLogicalAND<T, BinaryOperation>::value>;

template <typename T, class BinaryOperation>
using IsKnownIdentityOp =
    bool_constant<IsZeroIdentityOp<T, BinaryOperation>::value ||
                  IsOneIdentityOp<T, BinaryOperation>::value ||
                  IsOnesIdentityOp<T, BinaryOperation>::value ||
                  IsMinimumIdentityOp<T, BinaryOperation>::value ||
                  IsMaximumIdentityOp<T, BinaryOperation>::value ||
                  IsFalseIdentityOp<T, BinaryOperation>::value ||
                  IsTrueIdentityOp<T, BinaryOperation>::value>;

template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity_impl
    : std::integral_constant<
          bool, IsKnownIdentityOp<AccumulatorT, BinaryOperation>::value> {};

template <typename BinaryOperation, typename AccumulatorT, typename = void>
struct known_identity_impl {};

/// Returns zero as identity for ADD, OR, XOR operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<
    BinaryOperation, AccumulatorT,
    std::enable_if_t<IsZeroIdentityOp<AccumulatorT, BinaryOperation>::value>> {
  static constexpr AccumulatorT value = static_cast<AccumulatorT>(0);
};

#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, vec<std::byte, NumElements>,
    std::enable_if_t<IsZeroIdentityOp<vec<std::byte, NumElements>,
                                      BinaryOperation>::value>> {
  static constexpr vec<std::byte, NumElements> value =
      vec<std::byte, NumElements>(std::byte(0));
};

template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, marray<std::byte, NumElements>,
    std::enable_if_t<IsZeroIdentityOp<marray<std::byte, NumElements>,
                                      BinaryOperation>::value>> {
  static constexpr marray<std::byte, NumElements> value =
      marray<std::byte, NumElements>(std::byte(0));
};
#endif

template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, vec<sycl::half, NumElements>,
    std::enable_if_t<IsZeroIdentityOp<vec<sycl::half, NumElements>,
                                      BinaryOperation>::value>> {
  static constexpr vec<sycl::half, NumElements> value =
      vec<sycl::half, NumElements>(sycl::half());
};

template <typename BinaryOperation>
struct known_identity_impl<
    BinaryOperation, half,
    std::enable_if_t<IsZeroIdentityOp<half, BinaryOperation>::value>> {
  static constexpr half value =
#ifdef __SYCL_DEVICE_ONLY__
      0;
#else
      cl::sycl::detail::host_half_impl::half_v2(static_cast<uint16_t>(0));
#endif
};

/// Returns one as identify for MULTIPLY operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<
    BinaryOperation, AccumulatorT,
    std::enable_if_t<IsOneIdentityOp<AccumulatorT, BinaryOperation>::value>> {
  static constexpr AccumulatorT value = static_cast<AccumulatorT>(1);
};

#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, vec<std::byte, NumElements>,
    std::enable_if_t<
        IsOneIdentityOp<vec<std::byte, NumElements>, BinaryOperation>::value>> {
  static constexpr vec<std::byte, NumElements> value =
      vec<std::byte, NumElements>(std::byte(1));
};

template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, marray<std::byte, NumElements>,
    std::enable_if_t<IsOneIdentityOp<marray<std::byte, NumElements>,
                                     BinaryOperation>::value>> {
  static constexpr marray<std::byte, NumElements> value =
      marray<std::byte, NumElements>(std::byte(1));
};
#endif

template <typename BinaryOperation>
struct known_identity_impl<
    BinaryOperation, half,
    std::enable_if_t<IsOneIdentityOp<half, BinaryOperation>::value>> {
  static constexpr half value =
#ifdef __SYCL_DEVICE_ONLY__
      1;
#else
      cl::sycl::detail::host_half_impl::half_v2(static_cast<uint16_t>(0x3C00));
#endif
};

/// Returns bit image consisting of all ones as identity for AND operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<
    BinaryOperation, AccumulatorT,
    std::enable_if_t<IsOnesIdentityOp<AccumulatorT, BinaryOperation>::value>> {
  static constexpr AccumulatorT value = static_cast<AccumulatorT>(-1LL);
};

#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, vec<std::byte, NumElements>,
    std::enable_if_t<IsOnesIdentityOp<vec<std::byte, NumElements>,
                                      BinaryOperation>::value>> {
  static constexpr vec<std::byte, NumElements> value =
      vec<std::byte, NumElements>(std::byte(-1LL));
};

template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, marray<std::byte, NumElements>,
    std::enable_if_t<IsOnesIdentityOp<marray<std::byte, NumElements>,
                                      BinaryOperation>::value>> {
  static constexpr marray<std::byte, NumElements> value =
      marray<std::byte, NumElements>(std::byte(-1LL));
};
#endif

/// Returns maximal possible value as identity for MIN operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           std::enable_if_t<IsMinimumIdentityOp<
                               AccumulatorT, BinaryOperation>::value>> {
  static constexpr AccumulatorT value = static_cast<AccumulatorT>(
      std::numeric_limits<AccumulatorT>::has_infinity
          ? std::numeric_limits<AccumulatorT>::infinity()
          : (std::numeric_limits<AccumulatorT>::max)());
};

#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, vec<std::byte, NumElements>,
    std::enable_if_t<IsMinimumIdentityOp<vec<std::byte, NumElements>,
                                         BinaryOperation>::value>> {
  static constexpr vec<std::byte, NumElements> value =
      static_cast<vec<std::byte, NumElements>>(
          std::numeric_limits<vec<std::byte, NumElements>>::has_infinity
              ? std::numeric_limits<vec<std::byte, NumElements>>::infinity()
              : (std::numeric_limits<vec<std::byte, NumElements>>::max)());
};

template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, marray<std::byte, NumElements>,
    std::enable_if_t<IsMinimumIdentityOp<marray<std::byte, NumElements>,
                                         BinaryOperation>::value>> {
  static constexpr marray<std::byte, NumElements> value =
      static_cast<marray<std::byte, NumElements>>(
          std::numeric_limits<marray<std::byte, NumElements>>::has_infinity
              ? std::numeric_limits<marray<std::byte, NumElements>>::infinity()
              : (std::numeric_limits<marray<std::byte, NumElements>>::max)());
};
#endif

/// Returns minimal possible value as identity for MAX operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<BinaryOperation, AccumulatorT,
                           std::enable_if_t<IsMaximumIdentityOp<
                               AccumulatorT, BinaryOperation>::value>> {
  static constexpr AccumulatorT value = static_cast<AccumulatorT>(
      std::numeric_limits<AccumulatorT>::has_infinity
          ? static_cast<AccumulatorT>(
                -std::numeric_limits<AccumulatorT>::infinity())
          : std::numeric_limits<AccumulatorT>::lowest());
};

#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, vec<std::byte, NumElements>,
    std::enable_if_t<IsMaximumIdentityOp<vec<std::byte, NumElements>,
                                         BinaryOperation>::value>> {
  static constexpr vec<std::byte, NumElements> value = static_cast<
      vec<std::byte, NumElements>>(
      std::numeric_limits<vec<std::byte, NumElements>>::has_infinity
          ? static_cast<vec<std::byte, NumElements>>(
                -std::numeric_limits<vec<std::byte, NumElements>>::infinity())
          : std::numeric_limits<vec<std::byte, NumElements>>::lowest());
};

template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, marray<std::byte, NumElements>,
    std::enable_if_t<IsMaximumIdentityOp<marray<std::byte, NumElements>,
                                         BinaryOperation>::value>> {
  static constexpr marray<std::byte, NumElements> value =
      static_cast<marray<std::byte, NumElements>>(
          std::numeric_limits<marray<std::byte, NumElements>>::has_infinity
              ? static_cast<marray<std::byte, NumElements>>(
                    -std::numeric_limits<
                        marray<std::byte, NumElements>>::infinity())
              : std::numeric_limits<marray<std::byte, NumElements>>::lowest());
};
#endif

/// Returns false as identity for LOGICAL OR operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<
    BinaryOperation, AccumulatorT,
    std::enable_if_t<IsFalseIdentityOp<AccumulatorT, BinaryOperation>::value>> {
  static constexpr AccumulatorT value = static_cast<AccumulatorT>(false);
};

#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, vec<std::byte, NumElements>,
    std::enable_if_t<IsFalseIdentityOp<vec<std::byte, NumElements>,
                                       BinaryOperation>::value>> {
  static constexpr vec<std::byte, NumElements> value =
      vec<std::byte, NumElements>(std::byte(false));
};

template <typename BinaryOperation, size_t NumElements>
struct known_identity_impl<
    BinaryOperation, marray<std::byte, NumElements>,
    std::enable_if_t<IsFalseIdentityOp<marray<std::byte, NumElements>,
                                       BinaryOperation>::value>> {
  static constexpr marray<std::byte, NumElements> value =
      marray<std::byte, NumElements>(std::byte(false));
};
#endif

/// Returns true as identity for LOGICAL AND operations.
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity_impl<
    BinaryOperation, AccumulatorT,
    std::enable_if_t<IsTrueIdentityOp<AccumulatorT, BinaryOperation>::value>> {
  static constexpr AccumulatorT value = static_cast<AccumulatorT>(true);
};

#if __cplusplus >= 201703L && (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, vec<std::byte, NumElements>,
    std::enable_if_t<IsTrueIdentityOp<vec<std::byte, NumElements>,
                                      BinaryOperation>::value>> {
  static constexpr vec<std::byte, NumElements> value =
      vec<std::byte, NumElements>(std::byte(true));
};

template <typename BinaryOperation, int NumElements>
struct known_identity_impl<
    BinaryOperation, marray<std::byte, NumElements>,
    std::enable_if_t<IsTrueIdentityOp<marray<std::byte, NumElements>,
                                      BinaryOperation>::value>> {
  static constexpr marray<std::byte, NumElements> value =
      marray<std::byte, NumElements>(std::byte(true));
};
#endif

} // namespace detail

// ---- has_known_identity
template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity
    : detail::has_known_identity_impl<std::decay_t<BinaryOperation>,
                                      std::decay_t<AccumulatorT>> {};

template <typename BinaryOperation, typename AccumulatorT>
__SYCL_INLINE_CONSTEXPR bool has_known_identity_v =
    sycl::has_known_identity<BinaryOperation, AccumulatorT>::value;

// ---- known_identity
template <typename BinaryOperation, typename AccumulatorT>
struct known_identity
    : detail::known_identity_impl<std::decay_t<BinaryOperation>,
                                  std::decay_t<AccumulatorT>> {};

template <typename BinaryOperation, typename AccumulatorT>
__SYCL_INLINE_CONSTEXPR AccumulatorT known_identity_v =
    sycl::known_identity<BinaryOperation, AccumulatorT>::value;

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)