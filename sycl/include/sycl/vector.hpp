//==---------------- vector.hpp --- Implements sycl::vec -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Check if Clang's ext_vector_type attribute is available. Host compiler
// may not be Clang, and Clang may not be built with the extension.
#ifdef __clang__
#ifndef __has_extension
#define __has_extension(x) 0
#endif
#ifdef __HAS_EXT_VECTOR_TYPE__
#error "Undefine __HAS_EXT_VECTOR_TYPE__ macro"
#endif
#if __has_extension(attribute_ext_vector_type)
#define __HAS_EXT_VECTOR_TYPE__
#endif
#endif // __clang__

#if !defined(__HAS_EXT_VECTOR_TYPE__) && defined(__SYCL_DEVICE_ONLY__)
#error "SYCL device compiler is built without ext_vector_type support"
#endif

#include <sycl/access/access.hpp>              // for decorated, address_space
#include <sycl/aliases.hpp>                    // for half, cl_char, cl_int
#include <sycl/detail/common.hpp>              // for ArrayCreator, RepeatV...
#include <sycl/detail/defines_elementary.hpp>  // for __SYCL2020_DEPRECATED
#include <sycl/detail/generic_type_lists.hpp>  // for vector_basic_list
#include <sycl/detail/generic_type_traits.hpp> // for is_sigeninteger, is_s...
#include <sycl/detail/memcpy.hpp>              // for memcpy
#include <sycl/detail/named_swizzles_mixin.hpp>
#include <sycl/detail/type_list.hpp>      // for is_contained
#include <sycl/detail/type_traits.hpp>    // for is_floating_point
#include <sycl/detail/vector_convert.hpp> // for convertImpl
#include <sycl/detail/vector_traits.hpp>  // for vector_alignment
#include <sycl/half_type.hpp>             // for StorageT, half, Vec16...

#include <sycl/ext/oneapi/bfloat16.hpp> // bfloat16

#include <algorithm>   // for std::min
#include <array>       // for array
#include <cassert>     // for assert
#include <cstddef>     // for size_t, NULL, byte
#include <cstdint>     // for uint8_t, int16_t, int...
#include <functional>  // for divides, multiplies
#include <iterator>    // for pair
#include <ostream>     // for operator<<, basic_ost...
#include <type_traits> // for enable_if_t, is_same
#include <utility>     // for index_sequence, make_...

namespace sycl {
inline namespace _V1 {

struct elem {
  static constexpr int x = 0;
  static constexpr int y = 1;
  static constexpr int z = 2;
  static constexpr int w = 3;
  static constexpr int r = 0;
  static constexpr int g = 1;
  static constexpr int b = 2;
  static constexpr int a = 3;
  static constexpr int s0 = 0;
  static constexpr int s1 = 1;
  static constexpr int s2 = 2;
  static constexpr int s3 = 3;
  static constexpr int s4 = 4;
  static constexpr int s5 = 5;
  static constexpr int s6 = 6;
  static constexpr int s7 = 7;
  static constexpr int s8 = 8;
  static constexpr int s9 = 9;
  static constexpr int sA = 10;
  static constexpr int sB = 11;
  static constexpr int sC = 12;
  static constexpr int sD = 13;
  static constexpr int sE = 14;
  static constexpr int sF = 15;
};

namespace detail {
// Templated vs. non-templated conversion operator behaves differently when two
// conversions are needed as in the case below:
//
//   sycl::vec<int, 1> v;
//   std::ignore = static_cast<bool>(v);
//
// Make sure the snippet above compiles. That is important because
//
//   sycl::vec<int, 2> v;
//   if (v.x() == 42)
//     ...
//
// must go throw `v.x()` returning a swizzle, then its `operator==` returning
// vec<int, 1> and we want that code to compile.
template <typename Self, typename T, int N, typename = void>
struct ScalarConversionOperatorMixIn {};

template <typename Self, typename T, int N>
struct ScalarConversionOperatorMixIn<Self, T, N, std::enable_if_t<N == 1>> {
  operator T() const { return (*static_cast<const Self *>(this))[0]; }
};

// Everything could have been much easier if we had C++20 concepts, then all the
// operators could be provided in a single mixin class with proper `requires`
// clauses on each overload. Until then, we have to have at least a separate
// mixing for each requirement (e.g. not byte, neither byte nor fp, not fp,
// etc.). Grouping like that would also be somewhat confusing, so we just create
// a separate mixin for each overload/narrow set of overloads and just "merge"
// them all back later.

template <typename SelfOperandTy, typename DataT, bool EnablePostfix,
          typename = void>
struct IncDecMixin {};

template <typename SelfOperandTy, typename DataT>
struct IncDecMixin<SelfOperandTy, DataT, false,
                   std::enable_if_t<!std::is_same_v<bool, DataT>>> {
  friend SelfOperandTy &operator++(SelfOperandTy &x) {
    x += DataT{1};
    return x;
  }
  friend SelfOperandTy &operator--(SelfOperandTy &x) {
    x -= DataT{1};
    return x;
  }
};

template <typename SelfOperandTy, typename DataT>
struct IncDecMixin<SelfOperandTy, DataT, true,
                   std::enable_if_t<!std::is_same_v<bool, DataT>>>
    : public IncDecMixin<SelfOperandTy, DataT, false> {
  friend auto operator++(SelfOperandTy &x, int) {
    auto tmp = +x;
    x += DataT{1};
    return tmp;
  }
  friend auto operator--(SelfOperandTy &x, int) {
    auto tmp = +x;
    x -= DataT{1};
    return tmp;
  }
};

// TODO: The specification doesn't mention this specifically, but that's what
// the implementation has been doing and it seems to be a reasonable thing to
// do. Otherwise shift operators for byte element type would have to be disabled
// completely to follow C++ standard approach.
template <typename Self, typename OpAssignSelfOperandTy, typename DataT, int N,
          bool EnableOpAssign, typename = void>
struct ByteShiftsMixin {};

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <typename Self, typename OpAssignSelfOperandTy, typename DataT, int N>
struct ByteShiftsMixin<Self, OpAssignSelfOperandTy, DataT, N, false,
                       std::enable_if_t<std::is_same_v<std::byte, DataT>>> {
  friend auto operator<<(const Self &lhs, int shift) {
    vec<DataT, N> tmp;
    for (int i = 0; i < N; ++i)
      tmp[i] = lhs[i] << shift;
    return tmp;
  }
  friend auto operator>>(const Self &lhs, int shift) {
    vec<DataT, N> tmp;
    for (int i = 0; i < N; ++i)
      tmp[i] = lhs[i] >> shift;
    return tmp;
  }
};

template <typename Self, typename OpAssignSelfOperandTy, typename DataT, int N>
struct ByteShiftsMixin<Self, OpAssignSelfOperandTy, DataT, N, true,
                       std::enable_if_t<std::is_same_v<std::byte, DataT>>>
    : public ByteShiftsMixin<Self, OpAssignSelfOperandTy, DataT, N, false> {
  friend OpAssignSelfOperandTy &operator<<=(OpAssignSelfOperandTy &lhs,
                                            int shift) {
    lhs = lhs << shift;
    return lhs;
  }
  friend OpAssignSelfOperandTy &operator>>=(OpAssignSelfOperandTy &lhs,
                                            int shift) {
    lhs = lhs >> shift;
    return lhs;
  }
};
#endif

// We use std::plus<void> and similar to "map" template parameter to an
// overloaded operator. These three below are missing from `<functional>`.
struct ShiftLeft {
  template <class T, class U>
  constexpr auto operator()(T &&lhs, U &&rhs) const
      -> decltype(std::forward<T>(lhs) << std::forward<U>(rhs)) {
    return std::forward<T>(lhs) << std::forward<U>(rhs);
  }
};
struct ShiftRight {
  template <class T, class U>
  constexpr auto operator()(T &&lhs,
                            U &&rhs) const -> decltype(std::forward<T>(lhs) >>
                                                       std::forward<U>(rhs)) {
    return std::forward<T>(lhs) >> std::forward<U>(rhs);
  }
};

struct UnaryPlus {
  template <class T>
  constexpr auto operator()(T &&arg) const -> decltype(+std::forward<T>(arg)) {
    return +std::forward<T>(arg);
  }
};

template <class T>
static constexpr bool not_fp =
    !std::is_same_v<T, float> && !std::is_same_v<T, double> &&
    !std::is_same_v<T, half>;

// To provide information about operators availability depending on vec/swizzle
// element type.
template <typename Op, typename T>
inline constexpr bool is_op_available = false;

#define __SYCL_OP_AVAILABILITY(OP, COND)                                       \
  template <typename T> inline constexpr bool is_op_available<OP, T> = COND;

// clang-format off
__SYCL_OP_AVAILABILITY(std::plus<void>          , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::minus<void>         , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::multiplies<void>    , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::divides<void>       , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::modulus<void>       , !detail::is_byte_v<T> && not_fp<T>)

__SYCL_OP_AVAILABILITY(std::bit_and<void>       , not_fp<T>)
__SYCL_OP_AVAILABILITY(std::bit_or<void>        , not_fp<T>)
__SYCL_OP_AVAILABILITY(std::bit_xor<void>       , not_fp<T>)

__SYCL_OP_AVAILABILITY(std::equal_to<void>      , true)
__SYCL_OP_AVAILABILITY(std::not_equal_to<void>  , true)
__SYCL_OP_AVAILABILITY(std::less<void>          , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::greater<void>       , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::less_equal<void>    , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::greater_equal<void> , !detail::is_byte_v<T>)

__SYCL_OP_AVAILABILITY(std::logical_and<void>   , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::logical_or<void>    , !detail::is_byte_v<T>)

__SYCL_OP_AVAILABILITY(ShiftLeft     , !detail::is_byte_v<T> && not_fp<T>)
__SYCL_OP_AVAILABILITY(ShiftRight    , !detail::is_byte_v<T> && not_fp<T>)

// Unary
__SYCL_OP_AVAILABILITY(std::negate<void>       , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::logical_not<void>  , !detail::is_byte_v<T>)
__SYCL_OP_AVAILABILITY(std::bit_not<void>      , not_fp<T>)
__SYCL_OP_AVAILABILITY(UnaryPlus               , !detail::is_byte_v<T>)
// clang-format on

#undef __SYCL_OP_AVAILABILITY

// clang-format off
#define __SYCL_PROCESS_BINARY_OPS(PROCESS_OP, DELIMITER) \
          PROCESS_OP(std::plus<void>)          \
DELIMITER PROCESS_OP(std::minus<void>)         \
DELIMITER PROCESS_OP(std::multiplies<void>)    \
DELIMITER PROCESS_OP(std::divides<void>)       \
DELIMITER PROCESS_OP(std::modulus<void>)       \
DELIMITER PROCESS_OP(std::bit_and<void>)       \
DELIMITER PROCESS_OP(std::bit_or<void>)        \
DELIMITER PROCESS_OP(std::bit_xor<void>)       \
DELIMITER PROCESS_OP(std::equal_to<void>)      \
DELIMITER PROCESS_OP(std::not_equal_to<void>)  \
DELIMITER PROCESS_OP(std::less<void>)          \
DELIMITER PROCESS_OP(std::greater<void>)       \
DELIMITER PROCESS_OP(std::less_equal<void>)    \
DELIMITER PROCESS_OP(std::greater_equal<void>) \
DELIMITER PROCESS_OP(std::logical_and<void>)   \
DELIMITER PROCESS_OP(std::logical_or<void>)    \
DELIMITER PROCESS_OP(ShiftLeft)                \
DELIMITER PROCESS_OP(ShiftRight)

#define __SYCL_PROCESS_BINARY_OPASSIGN_OPS(PROCESS_OP, DELIMITER) \
          PROCESS_OP(std::plus<void>)          \
DELIMITER PROCESS_OP(std::minus<void>)         \
DELIMITER PROCESS_OP(std::multiplies<void>)    \
DELIMITER PROCESS_OP(std::divides<void>)       \
DELIMITER PROCESS_OP(std::modulus<void>)       \
DELIMITER PROCESS_OP(std::bit_and<void>)       \
DELIMITER PROCESS_OP(std::bit_or<void>)        \
DELIMITER PROCESS_OP(std::bit_xor<void>)       \
DELIMITER PROCESS_OP(ShiftLeft)                \
DELIMITER PROCESS_OP(ShiftRight)

#define __SYCL_PROCESS_UNARY_OPS(PROCESS_OP, DELIMITER) \
          PROCESS_OP(std::negate<void>)          \
DELIMITER PROCESS_OP(std::logical_not<void>)     \
DELIMITER PROCESS_OP(std::bit_not<void>)         \
DELIMITER PROCESS_OP(UnaryPlus)
// clang-format on

// Need to separate binop/opassign because const vec swizzles don't have the
// latter.
template <typename Lhs, typename Rhs, typename Impl, typename DataT,
          typename Op, typename = void>
struct NonTemplateBinaryOpMixin {};
template <typename Lhs, typename Rhs, typename DataT, typename Op,
          typename = void>
struct NonTemplateBinaryOpAssignMixin {};

template <typename VecT, int... Indexes> class __SYCL_EBO Swizzle;

template <typename Self, typename VecT, typename DataT, int N, typename Op,
          typename = void>
struct SwizzleTemplateBinaryOpMixin {};
template <typename Self, typename VecT, typename DataT, int N, typename Op,
          typename = void>
struct SwizzleTemplateBinaryOpAssignMixin {};

#define __SYCL_BINARY_OP_MIXIN(OP, BINOP)                                      \
  template <typename Lhs, typename Rhs, typename Impl, typename DataT>         \
  struct NonTemplateBinaryOpMixin<                                             \
      Lhs, Rhs, Impl, DataT, OP,                                               \
      std::enable_if_t<is_op_available<OP, DataT>>> {                          \
    friend auto operator BINOP(const Lhs &lhs, const Rhs &rhs) {               \
      return Impl{}(lhs, rhs, OP{});                                           \
    }                                                                          \
  };                                                                           \
  template <typename Self, typename VecT, typename DataT, int N>               \
  struct SwizzleTemplateBinaryOpMixin<                                         \
      Self, VecT, DataT, N, OP,                                                \
      std::enable_if_t<is_op_available<OP, DataT>>> {                          \
    template <typename OtherVecT, int... OtherIndexes,                         \
              typename = std::enable_if_t<                                     \
                  std::is_same_v<DataT, typename VecT::element_type> &&        \
                  N == sizeof...(OtherIndexes)>>                               \
    friend auto                                                                \
    operator BINOP(const Self &lhs,                                            \
                   const Swizzle<OtherVecT, OtherIndexes...> &rhs) {           \
      using ResultVec = vec<DataT, N>;                                         \
      return OP{}(static_cast<ResultVec>(lhs), static_cast<ResultVec>(rhs));   \
    }                                                                          \
    template <typename OtherVecT, int... OtherIndexes,                         \
              typename = std::enable_if_t<                                     \
                  std::is_same_v<DataT, typename VecT::element_type> &&        \
                  N == sizeof...(OtherIndexes) &&                              \
                  std::is_const_v<VecT> != std::is_const_v<OtherVecT>>>        \
    friend auto operator BINOP(const Swizzle<OtherVecT, OtherIndexes...> &lhs, \
                               const Self &rhs) {                              \
      using ResultVec = vec<DataT, N>;                                         \
      return OP{}(static_cast<ResultVec>(lhs), static_cast<ResultVec>(rhs));   \
    }                                                                          \
  };

#define __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(OP, BINOP, OPASSIGN)               \
  __SYCL_BINARY_OP_MIXIN(OP, BINOP)                                            \
  template <typename Lhs, typename Rhs, typename DataT>                        \
  struct NonTemplateBinaryOpAssignMixin<                                       \
      Lhs, Rhs, DataT, OP, std::enable_if_t<is_op_available<OP, DataT>>> {     \
    friend Lhs &operator OPASSIGN(Lhs & lhs, const Rhs & rhs) {                \
      lhs = OP{}(lhs, rhs);                                                    \
      return lhs;                                                              \
    }                                                                          \
  };                                                                           \
  template <typename Self, typename VecT, typename DataT, int N>               \
  struct SwizzleTemplateBinaryOpAssignMixin<                                   \
      Self, VecT, DataT, N, OP,                                                \
      std::enable_if_t<is_op_available<OP, DataT>>> {                          \
    template <typename OtherVecT, int... OtherIndexes,                         \
              typename = std::enable_if_t<                                     \
                  std::is_same_v<DataT, typename VecT::element_type> &&        \
                  N == sizeof...(OtherIndexes)>>                               \
    friend const Self &                                                        \
    operator OPASSIGN(const Self & lhs,                                        \
                      const Swizzle<OtherVecT, OtherIndexes...> &rhs) {        \
      using ResultVec = vec<DataT, N>;                                         \
      lhs = OP{}(static_cast<ResultVec>(lhs), static_cast<ResultVec>(rhs));    \
      return lhs;                                                              \
    }                                                                          \
    template <typename OtherVecT, int... OtherIndexes,                         \
              typename = std::enable_if_t<                                     \
                  std::is_same_v<DataT, typename VecT::element_type> &&        \
                  N == sizeof...(OtherIndexes) &&                              \
                  std::is_const_v<VecT> != std::is_const_v<OtherVecT>>>        \
    friend auto                                                                \
    operator OPASSIGN(const Swizzle<OtherVecT, OtherIndexes...> &lhs,          \
                      const Self &rhs) {                                       \
      using ResultVec = vec<DataT, N>;                                         \
      lhs = OP{}(static_cast<ResultVec>(lhs), static_cast<ResultVec>(rhs));    \
      return lhs;                                                              \
    }                                                                          \
  };

template <typename T, typename Impl, typename DataT, typename Op,
          typename = void>
struct UnaryOpMixin {};

#define __SYCL_UNARY_OP_MIXIN(OP, UOP)                                         \
  template <typename T, typename Impl, typename DataT>                         \
  struct UnaryOpMixin<T, Impl, DataT, OP,                                      \
                      std::enable_if_t<is_op_available<OP, DataT>>> {          \
    friend auto operator UOP(const T &x) { return Impl{}(x, OP{}); }           \
  };

// clang-format off
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::plus<void>       , +, +=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::minus<void>      , -, -=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::multiplies<void> , *, *=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::divides<void>    , /, /=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::modulus<void>    , %, %=)

  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::bit_and<void>    , &, &=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::bit_or<void>     , |, |=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::bit_xor<void>    , ^, ^=)

  __SYCL_BINARY_OP_MIXIN(std::equal_to<void>                , ==)
  __SYCL_BINARY_OP_MIXIN(std::not_equal_to<void>            , !=)
  __SYCL_BINARY_OP_MIXIN(std::less<void>                    , <)
  __SYCL_BINARY_OP_MIXIN(std::greater<void>                 , >)
  __SYCL_BINARY_OP_MIXIN(std::less_equal<void>              , <=)
  __SYCL_BINARY_OP_MIXIN(std::greater_equal<void>           , >=)

  __SYCL_BINARY_OP_MIXIN(std::logical_and<void>             , &&)
  __SYCL_BINARY_OP_MIXIN(std::logical_or<void>              , ||)

  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(ShiftLeft             , <<, <<=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(ShiftRight            , >>, >>=)

  __SYCL_UNARY_OP_MIXIN(std::negate<void>                   , -)
  __SYCL_UNARY_OP_MIXIN(std::logical_not<void>              , !)
  __SYCL_UNARY_OP_MIXIN(std::bit_not<void>                  , ~)
  __SYCL_UNARY_OP_MIXIN(UnaryPlus                           , +)
// clang-format on

#undef __SYCL_OP_MIXIN
#undef __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN
#undef __SYCL_BINARY_OP_MIXIN

#define __SYCL_COMMA ,

// clang-format off
#define __SYCL_MIXIN_FOR_BINARY(OP)                                            \
public NonTemplateBinaryOpMixin<Lhs, Rhs, Impl, DataT, OP>

#define __SYCL_MIXIN_FOR_BINARY_OPASSIGN(OP)                                   \
public NonTemplateBinaryOpAssignMixin<Lhs, Rhs, DataT, OP>

#define __SYCL_MIXIN_FOR_TEMPLATE_BINARY(OP)                                   \
public SwizzleTemplateBinaryOpMixin<Self, VecT, DataT, N, OP>

#define __SYCL_MIXIN_FOR_TEMPLATE_BINARY_OPASSIGN(OP)                          \
public SwizzleTemplateBinaryOpAssignMixin<Self, VecT, DataT, N, OP>

#define __SYCL_MIXIN_FOR_UNARY(OP)                                             \
public UnaryOpMixin<T, Impl, DataT, OP>
// clang-format on

template <typename Lhs, typename Rhs, typename Impl, typename DataT>
struct __SYCL_EBO NonTemplateBinaryOpsMixin
    : __SYCL_PROCESS_BINARY_OPS(__SYCL_MIXIN_FOR_BINARY, __SYCL_COMMA) {};

template <typename Lhs, typename Rhs, typename DataT>
struct __SYCL_EBO NonTemplateBinaryOpAssignOpsMixin
    : __SYCL_PROCESS_BINARY_OPASSIGN_OPS(__SYCL_MIXIN_FOR_BINARY_OPASSIGN,
                                         __SYCL_COMMA) {};

template <typename Self, typename VecT, typename DataT, int N>
struct __SYCL_EBO SwizzleTemplateBinaryOpsMixin
    : __SYCL_PROCESS_BINARY_OPS(__SYCL_MIXIN_FOR_TEMPLATE_BINARY,
                                __SYCL_COMMA) {};

template <typename Self, typename VecT, typename DataT, int N>
struct __SYCL_EBO SwizzleTemplateBinaryOpAssignOpsMixin
    : __SYCL_PROCESS_BINARY_OPASSIGN_OPS(
          __SYCL_MIXIN_FOR_TEMPLATE_BINARY_OPASSIGN, __SYCL_COMMA) {};

template <typename T, typename Impl, typename DataT>
struct __SYCL_EBO UnaryOpsMixin
    : __SYCL_PROCESS_UNARY_OPS(__SYCL_MIXIN_FOR_UNARY, __SYCL_COMMA) {};

#undef __SYCL_MIXIN_FOR_UNARY
#undef __SYCL_MIXIN_FOR_TEMPLATE_BINARY_OPASSIGN
#undef __SYCL_MIXIN_FOR_BINARY_OPASSIGN
#undef __SYCL_MIXIN_FOR_TEMPLATE_BINARY
#undef __SYCL_MIXIN_FOR_BINARY

#undef __SYCL_COMMA
#undef __SYCL_PROCESS_BINARY_OPS
#undef __SYCL_PROCESS_UNARY_OPS

struct SwizzleImpl {
private:
  template <typename T> static constexpr int num_elements() {
    if constexpr (is_vec_or_swizzle_v<T>)
      return T::size();
    else
      return 1;
  }

public:
  template <typename T0, typename T1, typename OpTy>
  auto operator()(const T0 &Lhs, const T1 &Rhs, OpTy &&Op) {
    static_assert(std::is_same_v<get_elem_type_t<T0>, get_elem_type_t<T1>>);
    constexpr auto N = (std::max)(num_elements<T0>(), num_elements<T1>());
    using ResultVec = vec<get_elem_type_t<T0>, N>;
    return Op(static_cast<ResultVec>(Lhs), static_cast<ResultVec>(Rhs));
  }
  template <typename T, typename OpTy> auto operator()(const T &X, OpTy &&Op) {
    using ResultVec = vec<typename T::element_type, T::size()>;
    return Op(static_cast<ResultVec>(X));
  }
};

struct VectorImpl {
private:
#ifdef __SYCL_DEVICE_ONLY__
  static constexpr bool is_host = false;
#else
  static constexpr bool is_host = true;
#endif

  template <typename T> static constexpr int num_elements() {
    if constexpr (is_vec_or_swizzle_v<T>)
      return T::size();
    else
      return 1;
  }

public:
  template <typename T0, typename T1, typename OpTy>
  auto operator()(const T0 &Lhs, const T1 &Rhs, OpTy &&Op) {
    static_assert(std::is_same_v<get_elem_type_t<T0>, get_elem_type_t<T1>>);
    constexpr auto N = (std::max)(num_elements<T0>(), num_elements<T1>());
    using DataT = get_elem_type_t<T0>;
    constexpr bool is_logical =
        std::is_same_v<OpTy, std::equal_to<void>> ||
        std::is_same_v<OpTy, std::not_equal_to<void>> ||
        std::is_same_v<OpTy, std::less<void>> ||
        std::is_same_v<OpTy, std::greater<void>> ||
        std::is_same_v<OpTy, std::less_equal<void>> ||
        std::is_same_v<OpTy, std::greater_equal<void>> ||
        std::is_same_v<OpTy, std::logical_and<void>> ||
        std::is_same_v<OpTy, std::logical_or<void>>;
    auto Get = [](const auto &a, int idx) {
      if constexpr (is_vec_v<std::remove_const_t<
                        std::remove_reference_t<decltype(a)>>>)
        return a[idx];
      else
        return a;
    };
    using ResultVec =
        vec<std::conditional_t<
                is_logical, detail::select_cl_scalar_integral_signed_t<DataT>,
                DataT>,
            N>;
    if constexpr (is_host || std::is_same_v<DataT, ext::oneapi::bfloat16> ||
                  std::is_same_v<DataT, bool> || N == 1) {
      // TODO: Optimized device impl for `is_logical`.
      ResultVec tmp{};
      for (int i = 0; i < N; ++i)
        if constexpr (is_logical)
          tmp[i] = Op(Get(Lhs, i), Get(Rhs, i)) ? -1 : 0;
        else
          tmp[i] = Op(Get(Lhs, i), Get(Rhs, i));
      return tmp;
    } else {
      using vec_t = vec<DataT, N>;
      using vector_t = typename vec_t::vector_t;
      if constexpr (is_logical) {
        // Workaround a crash in the C++ front end, reported internally.
        constexpr bool no_crash =
            std::is_same_v<OpTy, std::logical_and<void>> ||
            std::is_same_v<OpTy, std::logical_or<void>>;
        if constexpr (no_crash) {
          auto res = Op(static_cast<vector_t>(vec_t{Lhs}),
                        static_cast<vector_t>(vec_t{Rhs}));
          // bit_cast is needed to cast between char/signed char
          // `ext_vector_type`s.
          //
          // TODO: Can we just change `vector_t`, or is that some mismatch
          // between clang/SPIR-V?
          return ResultVec{sycl::bit_cast<typename ResultVec::vector_t>(res)};
        } else {
          auto vec_lhs = static_cast<vector_t>(vec_t{Lhs});
          auto vec_rhs = static_cast<vector_t>(vec_t{Rhs});
          auto res = [&]() {
            if constexpr (std::is_same_v<OpTy, std::equal_to<void>>)
              return vec_lhs == vec_rhs;
            else if constexpr (std::is_same_v<OpTy, std::not_equal_to<void>>)
              return vec_lhs != vec_rhs;
            else if constexpr (std::is_same_v<OpTy, std::less<void>>)
              return vec_lhs < vec_rhs;
            else if constexpr (std::is_same_v<OpTy, std::greater<void>>)
              return vec_lhs > vec_rhs;
            else if constexpr (std::is_same_v<OpTy, std::less_equal<void>>)
              return vec_lhs <= vec_rhs;
            else if constexpr (std::is_same_v<OpTy, std::greater_equal<void>>)
              return vec_lhs >= vec_rhs;
            else
              static_assert(!std::is_same_v<OpTy, OpTy>, "Must be unreachable");
          }();
          // See the comment above.
          return ResultVec{sycl::bit_cast<typename ResultVec::vector_t>(res)};
        }
      } else {
        return ResultVec{Op(static_cast<vector_t>(vec_t{Lhs}),
                            static_cast<vector_t>(vec_t{Rhs}))};
      }
    }
  }
  template <typename T, typename OpTy> auto operator()(const T &X, OpTy &&Op) {
    static_assert(is_vec_v<T>);
    constexpr bool is_logical = std::is_same_v<OpTy, std::logical_not<void>>;
    if constexpr (is_logical) {
      vec<detail::select_cl_scalar_integral_signed_t<typename T::element_type>,
          T::size()>
          tmp;
      for (int i = 0; i < T::size(); ++i)
        tmp[i] = Op(X[i]) ? -1 : 0;
      return tmp;
    } else if constexpr (is_host ||
                         std::is_same_v<bool, typename T::element_type>) {
      T tmp;
      for (int i = 0; i < T::size(); ++i)
        tmp[i] = Op(X[i]);
      return tmp;
    } else {
      return T{Op(static_cast<typename T::vector_t>(X))};
    }
  }
};

template <typename Self, typename VecT, typename DataT, int N,
          bool AllowAssignOps>
struct __SYCL_EBO SwizzleMixins
    : public NamedSwizzlesMixinConst<Self, N>,
      public NonTemplateBinaryOpsMixin<Self, DataT, SwizzleImpl, DataT>,
      public NonTemplateBinaryOpsMixin<DataT, Self, SwizzleImpl, DataT>,
      public NonTemplateBinaryOpsMixin<Self, vec<DataT, N>, SwizzleImpl, DataT>,
      public NonTemplateBinaryOpsMixin<vec<DataT, N>, Self, SwizzleImpl, DataT>,
      public UnaryOpsMixin<Self, SwizzleImpl, DataT>,
      public SwizzleTemplateBinaryOpsMixin<Self, VecT, DataT, N> {};

template <typename Self, typename VecT, typename DataT, int N>
struct __SYCL_EBO SwizzleMixins<Self, VecT, DataT, N, true>
    : public SwizzleMixins<Self, VecT, DataT, N, false>,
      public NonTemplateBinaryOpAssignOpsMixin<const Self, DataT, DataT>,
      public NonTemplateBinaryOpAssignOpsMixin<const Self, vec<DataT, N>,
                                               DataT>,
      // The next line isn't in the spec (yet?)
      public NonTemplateBinaryOpAssignOpsMixin<vec<DataT, N>, Self, DataT>,
      public SwizzleTemplateBinaryOpAssignOpsMixin<Self, VecT, DataT, N> {};

template <typename VecT, int... Indexes>
inline constexpr bool is_assignable_swizzle =
    !std::is_const_v<VecT> && []() constexpr {
      int Idxs[] = {Indexes...};
      for (std::size_t i = 1; i < sizeof...(Indexes); ++i) {
        for (std::size_t j = 0; j < i; ++j)
          if (Idxs[j] == Idxs[i])
            // Repeating index
            return false;
      }

      return true;
    }();

template <typename VecT, int... Indexes> class __SYCL_EBO Swizzle;

template <typename Self, typename DataT, int N, bool AllowAssignOps>
struct SwizzleBase {
  const Self &operator=(const Self &) = delete;
};

template <typename Self, typename DataT, int N>
struct SwizzleBase<Self, DataT, N, true> {
  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void load(size_t offset,
            multi_ptr<const DataT, AddressSpace, IsDecorated> ptr) const {
    vec<DataT, N> v;
    v.load(offset, ptr);
    *static_cast<Self *>(this) = v;
  }

  template <typename OtherVecT, int... OtherIndexes>
  std::enable_if_t<std::is_same_v<typename OtherVecT::element_type, DataT> &&
                       sizeof...(OtherIndexes) == N,
                   const Self &>
  operator=(const Swizzle<OtherVecT, OtherIndexes...> &rhs) {
    return (*this = static_cast<vec<DataT, N>>(rhs));
  }

  const Self &operator=(const vec<DataT, N> &rhs) const {
    for (int i = 0; i < N; ++i)
      (*static_cast<const Self *>(this))[i] = rhs[i];

    return *static_cast<const Self *>(this);
  }

  const Self &operator=(const DataT &rhs) const {
    for (int i = 0; i < N; ++i)
      (*static_cast<const Self *>(this))[i] = rhs;

    return *static_cast<const Self *>(this);
  }
};

template <typename VecT, int... Indexes>
class __SYCL_EBO Swizzle
    : public SwizzleBase<Swizzle<VecT, Indexes...>, typename VecT::element_type,
                         sizeof...(Indexes),
                         is_assignable_swizzle<VecT, Indexes...>>,
      public ScalarConversionOperatorMixIn<Swizzle<VecT, Indexes...>,
                                           typename VecT::element_type,
                                           sizeof...(Indexes)>,
      public IncDecMixin<const Swizzle<VecT, Indexes...>,
                         typename VecT::element_type,
                         is_assignable_swizzle<VecT, Indexes...>>,
      public ByteShiftsMixin<Swizzle<VecT, Indexes...>,
                             const Swizzle<VecT, Indexes...>,
                             typename VecT::element_type, sizeof...(Indexes),
                             is_assignable_swizzle<VecT, Indexes...>>,
      public SwizzleMixins<Swizzle<VecT, Indexes...>, VecT,
                           typename VecT::element_type, sizeof...(Indexes),
                           is_assignable_swizzle<VecT, Indexes...>> {
  using DataT = typename VecT::element_type;
  static constexpr int NumElements = sizeof...(Indexes);
  using ResultVec = vec<DataT, NumElements>;

  // Get underlying vec index for (*this)[idx] access.
  static constexpr auto get_vec_idx(int idx) {
    int counter = 0;
    int result = -1;
    ((result = counter++ == idx ? Indexes : result), ...);
    return result;
  }

public:
  using SwizzleBase<Swizzle<VecT, Indexes...>, typename VecT::element_type,
                    sizeof...(Indexes),
                    is_assignable_swizzle<VecT, Indexes...>>::operator=;

  using element_type = DataT;
  using value_type = DataT;

#ifdef __SYCL_DEVICE_ONLY__
  using vector_t = typename ResultVec::vector_t;
#endif

  Swizzle() = delete;
  Swizzle(const Swizzle &) = delete;

  explicit Swizzle(VecT &Vec) : Vec(Vec) {}

#ifdef __SYCL_DEVICE_ONLY__
  operator vector_t() const {
    return static_cast<vector_t>(static_cast<ResultVec>(*this));
  }
#endif

  static constexpr size_t byte_size() noexcept {
    return ResultVec::byte_size();
  }
  static constexpr size_t size() noexcept { return ResultVec::size(); }

  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  size_t get_size() const { return static_cast<ResultVec>(*this).get_size(); }

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const {
    return static_cast<ResultVec>(*this).get_count();
  };

  template <typename ConvertT,
            rounding_mode RoundingMode = rounding_mode::automatic>
  vec<ConvertT, NumElements> convert() const {
    return static_cast<ResultVec>(*this)
        .template convert<ConvertT, RoundingMode>();
  }

  template <typename asT> asT as() const {
    return static_cast<ResultVec>(*this).template as<asT>();
  }

  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void store(size_t offset,
             multi_ptr<DataT, AddressSpace, IsDecorated> ptr) const {
    return static_cast<ResultVec>(*this).store(offset, ptr);
  }

  operator ResultVec() const { return ResultVec{Vec[Indexes]...}; }

  template <int... swizzleIndexes> auto swizzle() const {
    return Vec.template swizzle<get_vec_idx(swizzleIndexes)...>();
  }

  auto &operator[](int index) const { return Vec[get_vec_idx(index)]; }

public:
  VecT &Vec;
};
} // namespace detail

///////////////////////// class sycl::vec /////////////////////////
// Provides a cross-platform vector class template that works efficiently on
// SYCL devices as well as in host C++ code.
template <typename DataT, int NumElements>
class __SYCL_EBO vec
    : public detail::ScalarConversionOperatorMixIn<vec<DataT, NumElements>,
                                                   DataT, NumElements>,
      public detail::IncDecMixin<vec<DataT, NumElements>, DataT,
                                 /* AllowAssignOps = */ true>,
      public detail::ByteShiftsMixin<
          vec<DataT, NumElements>, vec<DataT, NumElements>, DataT, NumElements,
          /* AllowAssignOps = */ true>,
      public detail::NamedSwizzlesMixinBoth<vec<DataT, NumElements>,
                                            NumElements>,
      public detail::NonTemplateBinaryOpsMixin<vec<DataT, NumElements>,
                                               vec<DataT, NumElements>,
                                               detail::VectorImpl, DataT>,
      public detail::NonTemplateBinaryOpsMixin<vec<DataT, NumElements>, DataT,
                                               detail::VectorImpl, DataT>,
      public detail::NonTemplateBinaryOpsMixin<DataT, vec<DataT, NumElements>,
                                               detail::VectorImpl, DataT>,
      public detail::UnaryOpsMixin<vec<DataT, NumElements>, detail::VectorImpl,
                                   DataT>,
      public detail::NonTemplateBinaryOpAssignOpsMixin<
          vec<DataT, NumElements>, vec<DataT, NumElements>, DataT>,
      public detail::NonTemplateBinaryOpAssignOpsMixin<vec<DataT, NumElements>,
                                                       DataT, DataT> {

  static_assert(NumElements == 1 || NumElements == 2 || NumElements == 3 ||
                    NumElements == 4 || NumElements == 8 || NumElements == 16,
                "Invalid number of elements for sycl::vec: only 1, 2, 3, 4, 8 "
                "or 16 are supported");
  static_assert(sizeof(bool) == sizeof(uint8_t), "bool size is not 1 byte");

  // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#memory-layout-and-alignment
  // It is required by the SPEC to align vec<DataT, 3> with vec<DataT, 4>.
  static constexpr size_t AdjustedNum = (NumElements == 3) ? 4 : NumElements;

  // This represent type of underlying value. There should be only one field
  // in the class, so vec<float, 16> should be equal to float16 in memory.
  using DataType = std::array<DataT, AdjustedNum>;

#ifdef __SYCL_DEVICE_ONLY__
  using element_type_for_vector_t = typename detail::map_type<
      DataT,
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
      std::byte, /*->*/ std::uint8_t, //
#endif
      bool, /*->*/ std::uint8_t,                            //
      sycl::half, /*->*/ sycl::detail::half_impl::StorageT, //
      sycl::ext::oneapi::bfloat16,
      /*->*/ sycl::ext::oneapi::detail::Bfloat16StorageT, //
      char, /*->*/ detail::ConvertToOpenCLType_t<char>,   //
      DataT, /*->*/ DataT                                 //
      >::type;

public:
  // Type used for passing sycl::vec to SPIRV builtins.
  // We can not use ext_vector_type(1) as it's not supported by SPIRV
  // plugins (CTS fails).
  using vector_t =
      typename std::conditional_t<NumElements == 1, element_type_for_vector_t,
                                  element_type_for_vector_t __attribute__((
                                      ext_vector_type(NumElements)))>;

private:
#endif // __SYCL_DEVICE_ONLY__

  static constexpr int getNumElements() { return NumElements; }

  // SizeChecker is needed for vec(const argTN &... args) ctor to validate
  // args.
  template <int Counter, int MaxValue, class...>
  struct SizeChecker : std::conditional_t<Counter == MaxValue, std::true_type,
                                          std::false_type> {};

  template <int Counter, int MaxValue, typename DataT_, class... tail>
  struct SizeChecker<Counter, MaxValue, DataT_, tail...>
      : std::conditional_t<Counter + 1 <= MaxValue,
                           SizeChecker<Counter + 1, MaxValue, tail...>,
                           std::false_type> {};

  // Utility trait for creating an std::array from an vector argument.
  template <typename DataT_, typename T, std::size_t... Is>
  static constexpr std::array<DataT_, sizeof...(Is)>
  VecToArray(const vec<T, sizeof...(Is)> &V, std::index_sequence<Is...>) {
    return {static_cast<DataT_>(V[Is])...};
  }
  template <typename DataT_, typename T, int N, int... T5, std::size_t... Is>
  static constexpr std::array<DataT_, sizeof...(Is)>
  VecToArray(const detail::Swizzle<vec<T, N>, T5...> &V,
             std::index_sequence<Is...>) {
    return {static_cast<DataT_>(V[Is])...};
  }
  template <typename DataT_, typename T, int N, int... T5, std::size_t... Is>
  static constexpr std::array<DataT_, sizeof...(Is)>
  VecToArray(const detail::Swizzle<const vec<T, N>, T5...> &V,
             std::index_sequence<Is...>) {
    return {static_cast<DataT_>(V[Is])...};
  }

  template <typename DataT_, typename T, int N>
  static constexpr std::array<DataT_, N>
  FlattenVecArgHelper(const vec<T, N> &A) {
    return VecToArray<DataT_>(A, std::make_index_sequence<N>());
  }
  template <typename DataT_, typename T, int N, int... Indexes_>
  static constexpr std::array<DataT_, sizeof...(Indexes_)>
  FlattenVecArgHelper(const detail::Swizzle<vec<T, N>, Indexes_...> &A) {
    return VecToArray<DataT_>(A,
                              std::make_index_sequence<sizeof...(Indexes_)>());
  }
  template <typename DataT_, typename T, int N, int... Indexes_>
  static constexpr std::array<DataT_, sizeof...(Indexes_)>
  FlattenVecArgHelper(const detail::Swizzle<const vec<T, N>, Indexes_...> &A) {
    return VecToArray<DataT_>(A,
                              std::make_index_sequence<sizeof...(Indexes_)>());
  }
  template <typename DataT_, typename T>
  static constexpr auto FlattenVecArgHelper(const T &A) {
    // static_cast required to avoid narrowing conversion warning
    // when T = unsigned long int and DataT_ = int.
    return std::array<DataT_, 1>{static_cast<DataT_>(A)};
  }
  template <typename DataT_, typename T> struct FlattenVecArg {
    constexpr auto operator()(const T &A) const {
      return FlattenVecArgHelper<DataT_>(A);
    }
  };

  // Alias for shortening the vec arguments to array converter.
  template <typename DataT_, typename... ArgTN>
  using VecArgArrayCreator =
      detail::ArrayCreator<DataT_, FlattenVecArg, ArgTN...>;

#define __SYCL_ALLOW_VECTOR_SIZES(num_elements)                                \
  template <int Counter, int MaxValue, typename DataT_, class... tail>         \
  struct SizeChecker<Counter, MaxValue, vec<DataT_, num_elements>, tail...>    \
      : std::conditional_t<                                                    \
            Counter + (num_elements) <= MaxValue,                              \
            SizeChecker<Counter + (num_elements), MaxValue, tail...>,          \
            std::false_type> {};                                               \
  template <int Counter, int MaxValue, typename DataT_, int... Indexes_,       \
            class... tail>                                                     \
  struct SizeChecker<Counter, MaxValue,                                        \
                     detail::Swizzle<vec<DataT_, num_elements>, Indexes_...>,  \
                     tail...>                                                  \
      : std::conditional_t<                                                    \
            Counter + sizeof...(Indexes_) <= MaxValue,                         \
            SizeChecker<Counter + sizeof...(Indexes_), MaxValue, tail...>,     \
            std::false_type> {};                                               \
  template <int Counter, int MaxValue, typename DataT_, int... Indexes_,       \
            class... tail>                                                     \
  struct SizeChecker<                                                          \
      Counter, MaxValue,                                                       \
      detail::Swizzle<const vec<DataT_, num_elements>, Indexes_...>, tail...>  \
      : std::conditional_t<                                                    \
            Counter + sizeof...(Indexes_) <= MaxValue,                         \
            SizeChecker<Counter + sizeof...(Indexes_), MaxValue, tail...>,     \
            std::false_type> {};

  __SYCL_ALLOW_VECTOR_SIZES(1)
  __SYCL_ALLOW_VECTOR_SIZES(2)
  __SYCL_ALLOW_VECTOR_SIZES(3)
  __SYCL_ALLOW_VECTOR_SIZES(4)
  __SYCL_ALLOW_VECTOR_SIZES(8)
  __SYCL_ALLOW_VECTOR_SIZES(16)
#undef __SYCL_ALLOW_VECTOR_SIZES

  // TypeChecker is needed for vec(const argTN &... args) ctor to validate
  // args.
  template <typename T, typename DataT_>
  struct TypeChecker : std::is_convertible<T, DataT_> {};
#define __SYCL_ALLOW_VECTOR_TYPES(num_elements)                                \
  template <typename DataT_>                                                   \
  struct TypeChecker<vec<DataT_, num_elements>, DataT_> : std::true_type {};   \
  template <typename DataT_, int... Indexes_>                                  \
  struct TypeChecker<detail::Swizzle<vec<DataT_, num_elements>, Indexes_...>,  \
                     DataT_> : std::true_type {};                              \
  template <typename DataT_, int... Indexes_>                                  \
  struct TypeChecker<                                                          \
      detail::Swizzle<const vec<DataT_, num_elements>, Indexes_...>, DataT_>   \
      : std::true_type {};

  __SYCL_ALLOW_VECTOR_TYPES(1)
  __SYCL_ALLOW_VECTOR_TYPES(2)
  __SYCL_ALLOW_VECTOR_TYPES(3)
  __SYCL_ALLOW_VECTOR_TYPES(4)
  __SYCL_ALLOW_VECTOR_TYPES(8)
  __SYCL_ALLOW_VECTOR_TYPES(16)
#undef __SYCL_ALLOW_VECTOR_TYPES

  template <int... Indexes> using Swizzle = detail::Swizzle<vec, Indexes...>;

  template <int... Indexes>
  using ConstSwizzle = detail::Swizzle<const vec, Indexes...>;

  // Shortcuts for args validation in vec(const argTN &... args) ctor.
  template <typename... argTN>
  using EnableIfSuitableTypes = typename std::enable_if_t<
      std::conjunction_v<TypeChecker<argTN, DataT>...>>;

  template <typename... argTN>
  using EnableIfSuitableNumElements =
      typename std::enable_if_t<SizeChecker<0, NumElements, argTN...>::value>;

  // Element type for relational operator return value.
  using rel_t = detail::select_cl_scalar_integral_signed_t<DataT>;

public:
  // Aliases required by SYCL 2020 to make sycl::vec consistent
  // with that of marray and buffer.
  using element_type = DataT;
  using value_type = DataT;

  /****************** Constructors **************/
  vec() = default;
  constexpr vec(const vec &Rhs) = default;
  constexpr vec(vec &&Rhs) = default;

private:
  // Implementation detail for the next public ctor.
  template <size_t... Is>
  constexpr vec(const std::array<DataT, NumElements> &Arr,
                std::index_sequence<Is...>)
      : m_Data{Arr[Is]...} {}

public:
  explicit constexpr vec(const DataT &arg)
      : vec{detail::RepeatValue<NumElements>(arg),
            std::make_index_sequence<NumElements>()} {}

  // Constructor from values of base type or vec of base type. Checks that
  // base types are match and that the NumElements == sum of lengths of args.
  template <typename... argTN, typename = EnableIfSuitableTypes<argTN...>,
            typename = EnableIfSuitableNumElements<argTN...>>
  constexpr vec(const argTN &...args)
      : vec{VecArgArrayCreator<DataT, argTN...>::Create(args...),
            std::make_index_sequence<NumElements>()} {}

  /****************** Assignment Operators **************/
  constexpr vec &operator=(const vec &Rhs) = default;

  // Template required to prevent ambiguous overload with the copy assignment
  // when NumElements == 1. The template prevents implicit conversion from
  // vec<_, 1> to DataT.
  template <typename Ty = DataT>
  typename std::enable_if_t<
      std::is_fundamental_v<Ty> ||
          detail::is_half_or_bf16_v<typename std::remove_const_t<Ty>>,
      vec &>
  operator=(const DataT &Rhs) {
    *this = vec{Rhs};
    return *this;
  }

  // W/o this, things like "vec<char,*> = vec<signed char, *>" doesn't work.
  template <typename Ty = DataT>
  typename std::enable_if_t<
      !std::is_same_v<Ty, rel_t> && std::is_convertible_v<Ty, rel_t>, vec &>
  operator=(const vec<rel_t, NumElements> &Rhs) {
    *this = Rhs.template as<vec>();
    return *this;
  }

#ifdef __SYCL_DEVICE_ONLY__
  // Make it a template to avoid ambiguity with `vec(const DataT &)` when
  // `vector_t` is the same as `DataT`. Not that the other ctor isn't a template
  // so we don't even need a smart `enable_if` condition here, the mere fact of
  // this being a template makes the other ctor preferred.
  template <
      typename vector_t_ = vector_t,
      typename = typename std::enable_if_t<std::is_same_v<vector_t_, vector_t>>>
  constexpr vec(vector_t_ openclVector) {
    m_Data = sycl::bit_cast<DataType>(openclVector);
  }

  /* @SYCL2020
   * Available only when: compiled for the device.
   * Converts this SYCL vec instance to the underlying backend-native vector
   * type defined by vector_t.
   */
  operator vector_t() const { return sycl::bit_cast<vector_t>(m_Data); }
#endif // __SYCL_DEVICE_ONLY__

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  static constexpr size_t get_count() { return size(); }
  static constexpr size_t size() noexcept { return NumElements; }
  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  static constexpr size_t get_size() { return byte_size(); }
  static constexpr size_t byte_size() noexcept { return sizeof(m_Data); }

private:
  // We interpret bool as int8_t, std::byte as uint8_t for conversion to other
  // types.
  template <typename T>
  using ConvertBoolAndByteT =
      typename detail::map_type<T,
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
                                std::byte, /*->*/ std::uint8_t, //
#endif
                                bool, /*->*/ std::uint8_t, //
                                T, /*->*/ T                //
                                >::type;

  // getValue should be able to operate on different underlying
  // types: enum cl_float#N , builtin vector float#N, builtin type float.
  constexpr auto getValue(int Index) const {
    using RetType =
        typename std::conditional_t<detail::is_byte_v<DataT>, int8_t,
#ifdef __SYCL_DEVICE_ONLY__
                                    element_type_for_vector_t
#else
                                    DataT
#endif
                                    >;

#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (std::is_same_v<DataT, sycl::ext::oneapi::bfloat16>)
      return sycl::bit_cast<RetType>(m_Data[Index]);
    else
#endif
      return static_cast<RetType>(m_Data[Index]);
  }

public:
  template <typename convertT,
            rounding_mode roundingMode = rounding_mode::automatic>
  vec<convertT, NumElements> convert() const {

    using T = ConvertBoolAndByteT<DataT>;
    using R = ConvertBoolAndByteT<convertT>;
    using bfloat16 = sycl::ext::oneapi::bfloat16;
    static_assert(std::is_integral_v<R> ||
                      detail::is_floating_point<R>::value ||
                      std::is_same_v<R, bfloat16>,
                  "Unsupported convertT");

    using OpenCLT = detail::ConvertToOpenCLType_t<T>;
    using OpenCLR = detail::ConvertToOpenCLType_t<R>;
    vec<convertT, NumElements> Result;

    // convertImpl can't be called with the same From and To types and therefore
    // we need some special processing in a few cases.
    if constexpr (std::is_same_v<DataT, convertT>) {
      return *this;
    } else if constexpr (std::is_same_v<OpenCLT, OpenCLR> ||
                         std::is_same_v<T, R>) {
      for (size_t I = 0; I < NumElements; ++I)
        Result[I] = static_cast<convertT>(getValue(I));
      return Result;
    } else {

#ifdef __SYCL_DEVICE_ONLY__
      using OpenCLVecT = OpenCLT __attribute__((ext_vector_type(NumElements)));
      using OpenCLVecR = OpenCLR __attribute__((ext_vector_type(NumElements)));

      auto NativeVector = sycl::bit_cast<vector_t>(*this);
      using ConvertTVecType = typename vec<convertT, NumElements>::vector_t;

      // Whole vector conversion can only be done, if:
      constexpr bool canUseNativeVectorConvert =
#ifdef __NVPTX__
          //  TODO: Likely unnecessary as
          //  https://github.com/intel/llvm/issues/11840 has been closed
          //  already.
          false &&
#endif
          NumElements > 1 &&
          // - vec storage has an equivalent OpenCL native vector it is
          //   implicitly convertible to. There are some corner cases where it
          //   is not the case with char, long and long long types.
          std::is_convertible_v<vector_t, OpenCLVecT> &&
          std::is_convertible_v<ConvertTVecType, OpenCLVecR> &&
          // - it is not a signed to unsigned (or vice versa) conversion
          //   see comments within 'convertImpl' for more details;
          !detail::is_sint_to_from_uint<T, R>::value &&
          // - destination type is not bool. bool is stored as integer under the
          //   hood and therefore conversion to bool looks like conversion
          //   between two integer types. Since bit pattern for true and false
          //   is not defined, there is no guarantee that integer conversion
          //   yields right results here;
          !std::is_same_v<convertT, bool>;

      if constexpr (canUseNativeVectorConvert) {
        auto val = detail::convertImpl<T, R, roundingMode, NumElements, OpenCLVecT,
                                OpenCLVecR>(NativeVector);
        Result.m_Data = sycl::bit_cast<decltype(Result.m_Data)>(val);
      } else
#endif // __SYCL_DEVICE_ONLY__
      {
        // Otherwise, we fallback to per-element conversion:
        for (size_t I = 0; I < NumElements; ++I) {
          auto val =
              detail::convertImpl<T, R, roundingMode, 1, OpenCLT, OpenCLR>(
                  getValue(I));
#ifdef __SYCL_DEVICE_ONLY__
          // On device, we interpret BF16 as uint16.
          if constexpr (std::is_same_v<convertT, bfloat16>)
            Result[I] = sycl::bit_cast<convertT>(val);
          else
#endif
            Result[I] = static_cast<convertT>(val);
        }
      }
    }
    return Result;
  }

  template <typename asT> asT as() const { return sycl::bit_cast<asT>(*this); }

  template <int... SwizzleIndexes> Swizzle<SwizzleIndexes...> swizzle() {
    return Swizzle<SwizzleIndexes...>{*this};
  }

  template <int... SwizzleIndexes>
  ConstSwizzle<SwizzleIndexes...> swizzle() const {
    return ConstSwizzle<SwizzleIndexes...>{*this};
  }

  const DataT &operator[](int i) const { return m_Data[i]; }

  DataT &operator[](int i) { return m_Data[i]; }

  template <access::address_space Space, access::decorated DecorateAddress>
  void load(size_t Offset, multi_ptr<const DataT, Space, DecorateAddress> Ptr) {
    for (int I = 0; I < NumElements; I++) {
      m_Data[I] = *multi_ptr<const DataT, Space, DecorateAddress>(
          Ptr + Offset * NumElements + I);
    }
  }
  template <access::address_space Space, access::decorated DecorateAddress>
  void load(size_t Offset, multi_ptr<DataT, Space, DecorateAddress> Ptr) {
    multi_ptr<const DataT, Space, DecorateAddress> ConstPtr(Ptr);
    load(Offset, ConstPtr);
  }
  template <int Dimensions, access::mode Mode,
            access::placeholder IsPlaceholder, access::target Target,
            typename PropertyListT>
  void
  load(size_t Offset,
       accessor<DataT, Dimensions, Mode, Target, IsPlaceholder, PropertyListT>
           Acc) {
    multi_ptr<const DataT, detail::TargetToAS<Target>::AS,
              access::decorated::yes>
        MultiPtr(Acc);
    load(Offset, MultiPtr);
  }
  void load(size_t Offset, const DataT *Ptr) {
    for (int I = 0; I < NumElements; ++I)
      m_Data[I] = Ptr[Offset * NumElements + I];
  }

  template <access::address_space Space, access::decorated DecorateAddress>
  void store(size_t Offset,
             multi_ptr<DataT, Space, DecorateAddress> Ptr) const {
    for (int I = 0; I < NumElements; I++) {
      *multi_ptr<DataT, Space, DecorateAddress>(Ptr + Offset * NumElements +
                                                I) = m_Data[I];
    }
  }
  template <int Dimensions, access::mode Mode,
            access::placeholder IsPlaceholder, access::target Target,
            typename PropertyListT>
  void
  store(size_t Offset,
        accessor<DataT, Dimensions, Mode, Target, IsPlaceholder, PropertyListT>
            Acc) {
    multi_ptr<DataT, detail::TargetToAS<Target>::AS, access::decorated::yes>
        MultiPtr(Acc);
    store(Offset, MultiPtr);
  }
  void store(size_t Offset, DataT *Ptr) const {
    for (int I = 0; I < NumElements; ++I)
      Ptr[Offset * NumElements + I] = m_Data[I];
  }

private:
  // fields
  // Alignment is the same as size, to a maximum size of 64. SPEC requires
  // "The elements of an instance of the SYCL vec class template are stored
  // in memory sequentially and contiguously and are aligned to the size of
  // the element type in bytes multiplied by the number of elements."
  static constexpr int alignment = (std::min)((size_t)64, sizeof(DataType));
  alignas(alignment) DataType m_Data;
  // friends
  template <typename T1, int T2> friend class __SYCL_EBO vec;
};
///////////////////////// class sycl::vec /////////////////////////

#ifdef __cpp_deduction_guides
// all compilers supporting deduction guides also support fold expressions
template <class T, class... U,
          class = std::enable_if_t<(std::is_same_v<T, U> && ...)>>
vec(T, U...) -> vec<T, sizeof...(U) + 1>;
#endif

} // namespace _V1
} // namespace sycl
