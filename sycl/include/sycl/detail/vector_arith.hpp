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
#include <sycl/detail/type_list.hpp>           // for is_contained
#include <sycl/detail/type_traits.hpp>         // for is_floating_point

#include <sycl/ext/oneapi/bfloat16.hpp> // bfloat16

#include <cstddef>
#include <type_traits> // for enable_if_t, is_same

namespace sycl {
inline namespace _V1 {

template <typename DataT, int NumElem> class vec;

namespace detail {

template <typename VecT> class VecAccess;

// Macros to populate binary operation on sycl::vec.
#if defined(__SYCL_BINOP) || defined(BINOP_BASE)
#error "Undefine __SYCL_BINOP and BINOP_BASE macro"
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define BINOP_BASE(BINOP, OPASSIGN, CONVERT, COND)                             \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_t> operator BINOP(const vec_t & Lhs,     \
                                                        const vec_t & Rhs) {   \
    vec_t Ret;                                                                 \
    if constexpr (vec_t::IsBfloat16) {                                         \
      for (size_t I = 0; I < NumElements; ++I) {                               \
        Ret[I] = Lhs[I] BINOP Rhs[I];                                          \
      }                                                                        \
    } else {                                                                   \
      auto ExtVecLhs = sycl::bit_cast<typename vec_t::vector_t>(Lhs);          \
      auto ExtVecRhs = sycl::bit_cast<typename vec_t::vector_t>(Rhs);          \
      Ret = vec<DataT, NumElements>(ExtVecLhs BINOP ExtVecRhs);                \
      if constexpr (std::is_same_v<DataT, bool> && CONVERT) {                  \
        vec_arith_common<bool, NumElements>::ConvertToDataT(Ret);              \
      }                                                                        \
    }                                                                          \
    return Ret;                                                                \
  }
#else // __SYCL_DEVICE_ONLY__

#define BINOP_BASE(BINOP, OPASSIGN, CONVERT, COND)                             \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec_t> operator BINOP(const vec_t & Lhs,     \
                                                        const vec_t & Rhs) {   \
    vec_t Ret{};                                                               \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret[I] = Lhs[I] BINOP Rhs[I];                                            \
    }                                                                          \
    return Ret;                                                                \
  }
#endif // __SYCL_DEVICE_ONLY__

#define __SYCL_BINOP(BINOP, OPASSIGN, CONVERT, COND)                           \
  BINOP_BASE(BINOP, OPASSIGN, CONVERT, COND)                                   \
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
  using ocl_t = detail::select_cl_scalar_integral_signed_t<DataT>;
  template <typename T> using vec_data = vec_helper<T>;

  // operator!.
  friend vec<ocl_t, NumElements> operator!(const vec_t &Rhs) {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (!vec_t::IsBfloat16) {
      auto extVec = sycl::bit_cast<typename vec_t::vector_t>(Rhs);
      vec<ocl_t, NumElements> Ret{
          (typename vec<ocl_t, NumElements>::vector_t) !extVec};
      return Ret;
    } else
#endif // __SYCL_DEVICE_ONLY__
    {
      vec<ocl_t, NumElements> Ret{};
      for (size_t I = 0; I < NumElements; ++I) {
        // static_cast will work here as the output of ! operator is either 0 or
        // -1.
        Ret[I] = static_cast<ocl_t>(-1 * (!Rhs[I]));
      }
      return Ret;
    }
  }

  // operator +.
  friend vec_t operator+(const vec_t &Lhs) {
#ifdef __SYCL_DEVICE_ONLY__
    auto extVec = sycl::bit_cast<typename vec_t::vector_t>(Lhs);
    return vec_t{+extVec};
#else
    vec_t Ret{};
    for (size_t I = 0; I < NumElements; ++I)
      Ret[I] = +Lhs[I];
    return Ret;
#endif
  }

  // operator -.
  friend vec_t operator-(const vec_t &Lhs) {
    vec_t Ret{};
    if constexpr (vec_t::IsBfloat16) {
      for (size_t I = 0; I < NumElements; I++)
        Ret[I] = -Lhs[I];
    } else {
#ifndef __SYCL_DEVICE_ONLY__
      for (size_t I = 0; I < NumElements; ++I)
        Ret[I] = -Lhs[I];
#else
      auto extVec = sycl::bit_cast<typename vec_t::vector_t>(Lhs);
      Ret = vec_t{-extVec};
      if constexpr (std::is_same_v<DataT, bool>) {
        vec_arith_common<bool, NumElements>::ConvertToDataT(Ret);
      }
#endif
    }
    return Ret;
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
#if defined(__SYCL_RELLOGOP) || defined(RELLOGOP_BASE)
#error "Undefine __SYCL_RELLOGOP and RELLOGOP_BASE macro."
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define RELLOGOP_BASE(RELLOGOP, COND)                                          \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec<ocl_t, NumElements>> operator RELLOGOP(  \
      const vec_t & Lhs, const vec_t & Rhs) {                                  \
    vec<ocl_t, NumElements> Ret{};                                             \
    /* ext_vector_type does not support bfloat16, so for these   */            \
    /* we do element-by-element operation on the underlying std::array.  */    \
    if constexpr (vec_t::IsBfloat16) {                                         \
      for (size_t I = 0; I < NumElements; ++I) {                               \
        Ret[I] = static_cast<ocl_t>(-(Lhs[I] RELLOGOP Rhs[I]));                \
      }                                                                        \
    } else {                                                                   \
      auto ExtVecLhs = sycl::bit_cast<typename vec_t::vector_t>(Lhs);          \
      auto ExtVecRhs = sycl::bit_cast<typename vec_t::vector_t>(Rhs);          \
      /* Cast required to convert unsigned char ext_vec_type to */             \
      /* char ext_vec_type. */                                                 \
      Ret = vec<ocl_t, NumElements>(                                           \
          (typename vec<ocl_t, NumElements>::vector_t)(                        \
              ExtVecLhs RELLOGOP ExtVecRhs));                                  \
      /* For NumElements == 1, we use scalar instead of ext_vector_type. */    \
      if constexpr (NumElements == 1) {                                        \
        Ret *= -1;                                                             \
      }                                                                        \
    }                                                                          \
    return Ret;                                                                \
  }
#else // __SYCL_DEVICE_ONLY__
#define RELLOGOP_BASE(RELLOGOP, COND)                                          \
  template <typename T = DataT>                                                \
  friend std::enable_if_t<(COND), vec<ocl_t, NumElements>> operator RELLOGOP(  \
      const vec_t & Lhs, const vec_t & Rhs) {                                  \
    vec<ocl_t, NumElements> Ret{};                                             \
    for (size_t I = 0; I < NumElements; ++I) {                                 \
      Ret[I] = static_cast<ocl_t>(-(Lhs[I] RELLOGOP Rhs[I]));                  \
    }                                                                          \
    return Ret;                                                                \
  }
#endif

#define __SYCL_RELLOGOP(RELLOGOP, COND)                                        \
  RELLOGOP_BASE(RELLOGOP, COND)                                                \
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
  __SYCL_RELLOGOP(==, true)
  __SYCL_RELLOGOP(!=, true)
  __SYCL_RELLOGOP(>, true)
  __SYCL_RELLOGOP(<, true)
  __SYCL_RELLOGOP(>=, true)
  __SYCL_RELLOGOP(<=, true)

  // Only available to integral types.
  __SYCL_RELLOGOP(&&, (!detail::is_vgenfloat_v<T>))
  __SYCL_RELLOGOP(||, (!detail::is_vgenfloat_v<T>))
#undef __SYCL_RELLOGOP
#undef RELLOGOP_BASE

  // Binary operations on sycl::vec<> for all types except std::byte.
  __SYCL_BINOP(+, +=, true, true)
  __SYCL_BINOP(-, -=, true, true)
  __SYCL_BINOP(*, *=, false, true)
  __SYCL_BINOP(/, /=, false, true)

  // The following OPs are available only when: DataT != cl_float &&
  // DataT != cl_double && DataT != cl_half && DataT != BF16.
  __SYCL_BINOP(%, %=, false, (!detail::is_vgenfloat_v<T>))
  // Bitwise operations are allowed for std::byte.
  __SYCL_BINOP(|, |=, false, (!detail::is_vgenfloat_v<DataT>))
  __SYCL_BINOP(&, &=, false, (!detail::is_vgenfloat_v<DataT>))
  __SYCL_BINOP(^, ^=, false, (!detail::is_vgenfloat_v<DataT>))
  __SYCL_BINOP(>>, >>=, false, (!detail::is_vgenfloat_v<DataT>))
  __SYCL_BINOP(<<, <<=, true, (!detail::is_vgenfloat_v<DataT>))

  // friends
  template <typename T1, int T2> friend class vec;
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

  __SYCL_BINOP(|, |=, false, true)
  __SYCL_BINOP(&, &=, false, true)
  __SYCL_BINOP(^, ^=, false, true)

  // friends
  template <typename T1, int T2> friend class vec;
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
#ifdef __SYCL_DEVICE_ONLY__
    auto extVec = sycl::bit_cast<typename vec_t::vector_t>(Rhs);
    vec_t Ret{~extVec};
    if constexpr (std::is_same_v<DataT, bool>) {
      ConvertToDataT(Ret);
    }
    return Ret;
#else
    vec_t Ret{};
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = ~Rhs[I];
    }
    return Ret;
#endif
  }

#ifdef __SYCL_DEVICE_ONLY__
  using vec_bool_t = vec<bool, NumElements>;
  // Required only for std::bool.
  static void ConvertToDataT(vec_bool_t &Ret) {
    for (size_t I = 0; I < NumElements; ++I) {
      Ret[I] = bit_cast<int8_t>(Ret[I]) != 0;
    }
  }
#endif

  // friends
  template <typename T1, int T2> friend class vec;
};

#undef __SYCL_BINOP
#undef BINOP_BASE

} // namespace detail
} // namespace _V1
} // namespace sycl
