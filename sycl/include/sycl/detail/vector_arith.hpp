//=== vector_arith.hpp --- Implementation of arithmetic ops on sycl::vec  ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/aliases.hpp>                    // for half, cl_char, cl_int
#include <sycl/detail/generic_type_traits.hpp> // for is_sigeninteger, is_s...
#include <sycl/detail/type_traits.hpp>         // for is_floating_point

#include <sycl/ext/oneapi/bfloat16.hpp> // bfloat16

#include <cstddef>
#include <type_traits> // for enable_if_t, is_same

namespace sycl {
inline namespace _V1 {

template <typename DataT, int NumElem> class __SYCL_EBO vec;

namespace detail {

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
  constexpr auto operator()(T &&lhs, U &&rhs) const
      -> decltype(std::forward<T>(lhs) >> std::forward<U>(rhs)) {
    return std::forward<T>(lhs) >> std::forward<U>(rhs);
  }
};

struct UnaryPlus {
  template <class T>
  constexpr auto operator()(T &&arg) const -> decltype(+std::forward<T>(arg)) {
    return +std::forward<T>(arg);
  }
};

struct VecOperators {
#ifdef __SYCL_DEVICE_ONLY__
  static constexpr bool is_host = false;
#else
  static constexpr bool is_host = true;
#endif

  template <typename BinOp, typename Self>
  static constexpr auto apply_binary_op(const Self &Lhs, const Self &Rhs) {
    static_assert(is_vec_v<Self>);
    using element_type = typename Self::element_type;
    constexpr int N = Self::size();
    constexpr bool is_logical =
        check_type_in_v<BinOp, std::equal_to<void>, std::not_equal_to<void>,
                        std::less<void>, std::greater<void>,
                        std::less_equal<void>, std::greater_equal<void>,
                        std::logical_and<void>, std::logical_or<void>>;
    using result_t = std::conditional_t<
        is_logical, vec<fixed_width_signed<sizeof(element_type)>, N>, Self>;

    BinOp Op{};
    if constexpr (is_host ||
                  std::is_same_v<element_type, ext::oneapi::bfloat16>) {
      result_t res;
      for (size_t i = 0; i < N; ++i)
        if constexpr (is_logical)
          res[i] = Op(Lhs[i], Rhs[i]) ? -1 : 0;
        else
          res[i] = Op(Lhs[i], Rhs[i]);
      return res;
    } else {
      using vector_t = typename Self::vector_t;
      auto res = [&](auto x, auto y) {
        // Workaround for https://godbolt.org/z/P6daoG6jW.
        if constexpr (std::is_same_v<BinOp, std::equal_to<void>>)
          return x == y;
        else if constexpr (std::is_same_v<BinOp, std::not_equal_to<void>>)
          return x != y;
        else if constexpr (std::is_same_v<BinOp, std::less<void>>)
          return x < y;
        else if constexpr (std::is_same_v<BinOp, std::less_equal<void>>)
          return x <= y;
        else if constexpr (std::is_same_v<BinOp, std::greater<void>>)
          return x > y;
        else if constexpr (std::is_same_v<BinOp, std::greater_equal<void>>)
          return x >= y;
        else
          return Op(x, y);
      }(bit_cast<vector_t>(Lhs), bit_cast<vector_t>(Rhs));
      if constexpr (is_logical) {
        if constexpr (N == 1) {
          // vector_t for one-element vector is just scalar with normal C++
          // semantics, need to align it with vector behavior:
          return result_t{(typename result_t::vector_t)res * -1};
        }
        return result_t{(typename result_t::vector_t)res};
      } else {
        if constexpr (std::is_same_v<element_type, bool>) {
          // OpenCL/SPIR-V has different semantics for operations on bool (which
          // is mapped to some 8-bit integer type on device) than SYCL. Need to
          // "convert" back to 0/1.
          //
          // Some operations are known to produce only 0/1 for valid bool inputs
          // though, no reason to do extra processing for them:
          if constexpr (!check_type_in_v<BinOp, std::multiplies<void>,
                                         std::divides<void>, std::bit_or<void>,
                                         std::bit_and<void>, std::bit_xor<void>,
                                         ShiftRight>) {
            for (size_t i = 0; i < N; ++i)
              res[i] = bit_cast<int8_t>(res[i]) != 0;
          }
        }
        return result_t{(typename result_t::vector_t)res};
      }
    }
  }

  template <typename UnaryOp, typename Self>
  static constexpr auto apply_unary_op(const Self &self) {
    static_assert(is_vec_v<Self>);
    constexpr bool is_logical = std::is_same_v<UnaryOp, std::logical_not<void>>;
    using element_type = typename Self::element_type;
    constexpr int N = Self::size();
    using result_t = std::conditional_t<
        is_logical, vec<fixed_width_signed<sizeof(element_type)>, N>, Self>;
    UnaryOp Op{};
    if constexpr (is_host ||
                  std::is_same_v<element_type, ext::oneapi::bfloat16>) {
      result_t res;
      for (size_t i = 0; i < N; ++i)
        if constexpr (is_logical)
          res[i] = Op(self[i]) ? -1 : 0;
        else
          res[i] = Op(self[i]);
      return res;
    } else {
      using vector_t = typename Self::vector_t;
      auto res = Op(bit_cast<vector_t>(self));
      if constexpr (is_logical) {
        if constexpr (N == 1) {
          // vector_t for one-element vector is just scalar with normal C++
          // semantics, need to align it with vector behavior:
          return result_t{(typename result_t::vector_t)res * -1};
        }
        return result_t{(typename result_t::vector_t)res};
      } else {
        if constexpr (std::is_same_v<element_type, bool>) {
          // OpenCL/SPIR-V has different semantics for operations on bool (which
          // is mapped to some 8-bit integer type on device) than SYCL. Need to
          // "convert" back to 0/1.
          //
          // Some operations are known to produce only 0/1 for valid bool inputs
          // though, no reason to do extra processing for them:
          if constexpr (!check_type_in_v<UnaryOp, UnaryPlus>) {
            for (size_t i = 0; i < N; ++i)
              res[i] = bit_cast<int8_t>(res[i]) != 0;
          }
        }
        return result_t{(typename result_t::vector_t)res};
      }
    }
  }
};

// Macros to populate binary operation on sycl::vec.
#if defined(__SYCL_BINOP)
#error "Undefine __SYCL_BINOP macro"
#endif

#define __SYCL_BINOP(BINOP, OPASSIGN, CONVERT, COND, FUNCTOR)                  \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_t> operator BINOP(const vec_t & Lhs,     \
                                                        const vec_t & Rhs) {   \
    return VecOperators::apply_binary_op<FUNCTOR>(Lhs, Rhs);                   \
  }                                                                            \
                                                                               \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_t> operator BINOP(const vec_t & Lhs,     \
                                                        const DataT & Rhs) {   \
    return Lhs BINOP vec_t(Rhs);                                               \
  }                                                                            \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_t> operator BINOP(const DataT & Lhs,     \
                                                        const vec_t & Rhs) {   \
    return vec_t(Lhs) BINOP Rhs;                                               \
  }                                                                            \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_t> &operator OPASSIGN(                   \
      vec_t & Lhs, const vec_t & Rhs) {                                        \
    Lhs = Lhs BINOP Rhs;                                                       \
    return Lhs;                                                                \
  }                                                                            \
  template <int Num = NumElements, typename T = DataT>                         \
  friend std::enable_if_t<(Num != 1) && (COND), vec_t &> operator OPASSIGN(    \
      vec_t & Lhs, const DataT & Rhs) {                                        \
    Lhs = Lhs BINOP vec_t(Rhs);                                                \
    return Lhs;                                                                \
  }

/****************************************************************
 *                       vec_arith_common
 *                 /           |             \
 *                /            |               \
 *     vec_arith<int>     vec_arith<float> ...   vec_arith<byte>
 *                \            |               /
 *                 \           |              /
 *                        sycl::vec<T>
 *
 * vec_arith_common is the base class for vec_arith. It contains
 * the common math operators of sycl::vec for all types.
 * vec_arith is the derived class that contains the math operators
 * specialized for certain types. sycl::vec inherits from vec_arith.
 * *************************************************************/
template <typename DataT, int NumElements> class vec_arith_common;
template <typename DataT> struct vec_helper;

template <typename DataT, int NumElements>
class vec_arith : public vec_arith_common<DataT, NumElements> {
protected:
  using vec_t = vec<DataT, NumElements>;
  using ocl_t = detail::fixed_width_signed<sizeof(DataT)>;
  template <typename T> using vec_data = vec_helper<T>;

  // operator!.
  friend vec<ocl_t, NumElements> operator!(const vec_t &Rhs) {
    return VecOperators::apply_unary_op<std::logical_not<void>>(Rhs);
  }

  // operator +.
  friend vec_t operator+(const vec_t &Lhs) {
    return VecOperators::apply_unary_op<UnaryPlus>(Lhs);
  }

  // operator -.
  friend vec_t operator-(const vec_t &Lhs) {
    return VecOperators::apply_unary_op<std::negate<void>>(Lhs);
  }

// Unary operations on sycl::vec
// FIXME: Don't allow Unary operators on vec<bool> after
// https://github.com/KhronosGroup/SYCL-CTS/issues/896 gets fixed.
#ifdef __SYCL_UOP
#error "Undefine __SYCL_UOP macro"
#endif
#define __SYCL_UOP(UOP, OPASSIGN)                                              \
  friend vec_t &operator UOP(vec_t & Rhs) {                                    \
    Rhs OPASSIGN DataT{1};                                                     \
    return Rhs;                                                                \
  }                                                                            \
  friend vec_t operator UOP(vec_t &Lhs, int) {                                 \
    vec_t Ret(Lhs);                                                            \
    Lhs OPASSIGN DataT{1};                                                     \
    return Ret;                                                                \
  }

  __SYCL_UOP(++, +=)
  __SYCL_UOP(--, -=)
#undef __SYCL_UOP

  // The logical operations on scalar types results in 0/1, while for vec<>,
  // logical operations should result in 0 and -1 (similar to OpenCL vectors).
  // That's why, for vec<DataT, 1>, we need to invert the result of the logical
  // operations since we store vec<DataT, 1> as scalar type on the device.
#if defined(__SYCL_RELLOGOP)
#error "Undefine __SYCL_RELLOGOP macro."
#endif

#define __SYCL_RELLOGOP(RELLOGOP, COND, FUNCTOR)                               \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec<ocl_t, NumElements>> operator RELLOGOP(  \
      const vec_t & Lhs, const vec_t & Rhs) {                                  \
    return VecOperators::apply_binary_op<FUNCTOR>(Lhs, Rhs);                   \
  }                                                                            \
                                                                               \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec<ocl_t, NumElements>> operator RELLOGOP(  \
      const vec_t & Lhs, const DataT & Rhs) {                                  \
    return Lhs RELLOGOP vec_t(Rhs);                                            \
  }                                                                            \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec<ocl_t, NumElements>> operator RELLOGOP(  \
      const DataT & Lhs, const vec_t & Rhs) {                                  \
    return vec_t(Lhs) RELLOGOP Rhs;                                            \
  }

  // OP is: ==, !=, <, >, <=, >=, &&, ||
  // vec<RET, NumElements> operatorOP(const vec<DataT, NumElements> &Rhs) const;
  // vec<RET, NumElements> operatorOP(const DataT &Rhs) const;
  __SYCL_RELLOGOP(==, true, std::equal_to<void>)
  __SYCL_RELLOGOP(!=, true, std::not_equal_to<void>)
  __SYCL_RELLOGOP(>, true, std::greater<void>)
  __SYCL_RELLOGOP(<, true, std::less<void>)
  __SYCL_RELLOGOP(>=, true, std::greater_equal<void>)
  __SYCL_RELLOGOP(<=, true, std::less_equal<void>)

  // Only available to integral types.
  __SYCL_RELLOGOP(&&, (!detail::is_vgenfloat_v<T>), std::logical_and<void>)
  __SYCL_RELLOGOP(||, (!detail::is_vgenfloat_v<T>), std::logical_or<void>)
#undef __SYCL_RELLOGOP
#undef RELLOGOP_BASE

  // clang-format off
  // Binary operations on sycl::vec<> for all types except std::byte.
  __SYCL_BINOP(+, +=, true, true, std::plus<void>)
  __SYCL_BINOP(-, -=, true, true, std::minus<void>)
  __SYCL_BINOP(*, *=, false, true, std::multiplies<void>)
  __SYCL_BINOP(/, /=, false, true, std::divides<void>)

  // The following OPs are available only when: DataT != cl_float &&
  // DataT != cl_double && DataT != cl_half && DataT != BF16.
  __SYCL_BINOP(%, %=, false, (!detail::is_vgenfloat_v<T>), std::modulus<void>)
  // Bitwise operations are allowed for std::byte.
  __SYCL_BINOP(|, |=, false, (!detail::is_vgenfloat_v<DataT>), std::bit_or<void>)
  __SYCL_BINOP(&, &=, false, (!detail::is_vgenfloat_v<DataT>), std::bit_and<void>)
  __SYCL_BINOP(^, ^=, false, (!detail::is_vgenfloat_v<DataT>), std::bit_xor<void>)
  __SYCL_BINOP(>>, >>=, false, (!detail::is_vgenfloat_v<DataT>), ShiftRight)
  __SYCL_BINOP(<<, <<=, true, (!detail::is_vgenfloat_v<DataT>), ShiftLeft)
  // clang-format on

  // friends
  template <typename T1, int T2> friend class __SYCL_EBO vec;
}; // class vec_arith<>

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <int NumElements>
class vec_arith<std::byte, NumElements>
    : public vec_arith_common<std::byte, NumElements> {
protected:
  // NumElements can never be zero. Still using the redundant check to avoid
  // incomplete type errors.
  using DataT = typename std::conditional_t<NumElements == 0, int, std::byte>;
  using vec_t = vec<DataT, NumElements>;
  template <typename T> using vec_data = vec_helper<T>;

  // Special <<, >> operators for std::byte.
  // std::byte is not an arithmetic type and it only supports the following
  // overloads of >> and << operators.
  //
  // 1 template <class IntegerType>
  //   constexpr std::byte operator<<( std::byte b, IntegerType shift )
  //   noexcept;
  friend vec_t operator<<(const vec_t &Lhs, int shift) {
    vec_t Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = Lhs[I] << shift;
    }
    return Ret;
  }
  friend vec_t &operator<<=(vec_t &Lhs, int shift) {
    Lhs = Lhs << shift;
    return Lhs;
  }

  // 2 template <class IntegerType>
  //   constexpr std::byte operator>>( std::byte b, IntegerType shift )
  //   noexcept;
  friend vec_t operator>>(const vec_t &Lhs, int shift) {
    vec_t Ret;
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = Lhs[I] >> shift;
    }
    return Ret;
  }
  friend vec_t &operator>>=(vec_t &Lhs, int shift) {
    Lhs = Lhs >> shift;
    return Lhs;
  }

  __SYCL_BINOP(|, |=, false, true, std::bit_or<void>)
  __SYCL_BINOP(&, &=, false, true, std::bit_and<void>)
  __SYCL_BINOP(^, ^=, false, true, std::bit_xor<void>)

  // friends
  template <typename T1, int T2> friend class __SYCL_EBO vec;
};
#endif // (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)

template <typename DataT, int NumElements> class vec_arith_common {
protected:
  using vec_t = vec<DataT, NumElements>;

  static constexpr bool IsBfloat16 =
      std::is_same_v<DataT, sycl::ext::oneapi::bfloat16>;

  // operator~() available only when: dataT != float && dataT != double
  // && dataT != half
  template <typename T = DataT>
  friend std::enable_if_t<!detail::is_vgenfloat_v<T>, vec_t>
  operator~(const vec_t &Rhs) {
    return VecOperators::apply_unary_op<std::bit_not<void>>(Rhs);
  }

  // friends
  template <typename T1, int T2> friend class __SYCL_EBO vec;
};

#undef __SYCL_BINOP

} // namespace detail
} // namespace _V1
} // namespace sycl
