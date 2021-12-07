//==------------ - elem_type_traits.hpp - DPC++ Explicit SIMD API ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header provides:
// - meta interfaces ("traits") which must be implemented to supoprt
//   non-standard element types, such as sycl::half or sycl::bfloat16
// - interfaces for performing various C++ operations on the types (+, *,...)
//   and their default implementations.
// - interfaces for performing conversions to/from the types and their default
//   implementations.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>

#include <CL/sycl/half_type.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {
namespace detail {

enum class BinOp {
  ARITH_FIRST,
  add = ARITH_FIRST,
  sub,
  mul,
  div,
  rem,
  ARITH_LAST = rem,
  BIT_FIRST,
  shl = BIT_FIRST,
  shr,
  BIT_LOG,
  bit_or = BIT_LOG,
  bit_and,
  bit_xor,
  BIT_LST = bit_xor,
  LOG_FIRST,
  log_or = LOG_FIRST,
  log_and,
  LOG_LAST = log_and
};

enum class CmpOp {
  CMP_FIRST,
  lt = CMP_FIRST,
  lte,
  gte,
  gt,
  EQ_CMP_FIRST,
  eq = EQ_CMP_FIRST,
  ne,
  CMP_LAST = ne
};

// If this element type is a special "wrapper" type. Such wrapper types, e.g.
// sycl::half, bfloat etc, are used to represent non-standard element types in
// user code.
template <class T>
static inline constexpr bool is_wrapper_elem_type_v =
    std::is_same_v<T, sycl::half>;

template <class T>
static inline constexpr bool is_valid_simd_elem_type_v =
    (is_vectorizable_v<T> || is_wrapper_elem_type_v<T>);

struct invalid_storage_element_type;

template <class T, class SFINAE> struct element_type_traits {
  // The raw element type of the underlying clang vector used as a
  // storage.
  using StorageT = invalid_storage_element_type;
  // A starndard C++ type which this one can be converted to/from.
  // The conversions are usually H/W-supported, and the C++ type can
  // represent the entire range of values of this type.
  using EnclosingCppT = void;
  // Whether a value or clang vector value the raw element type can be used
  // directly as operand to std C++ operations.
  static inline constexpr bool use_native_cpp_ops = true;
};

template <class T>
struct element_type_traits<T, std::enable_if_t<is_vectorizable_v<T>>> {
  using StorageT = T;
  using EnclosingCppT = T;
  static inline constexpr bool use_native_cpp_ops = true;
};

template <class T>
using element_storage_t = typename element_type_traits<T>::StorageT;

// --- Type conversions

// Low-level conversion functions to and from a wrapper ("user") element type.
// Must be implemented for each supported
// <wrapper element type, C++ std type pair>.

// These are default implementations for wrapper types with native cpp
// operations support for their corresponding storage type.
template <class WrapperTy, class StdTy, int N>
vector_type_t<element_storage_t<WrapperTy>, N>
__esimd_convertvector_to(vector_type_t<StdTy, N> Val)
#ifdef __SYCL_DEVICE_ONLY__
    ; // needs to be implemented for WrapperTy's for which
      // element_type_traits<WrapperTy>::use_native_cpp_ops is false.
#else
{
  // TODO implement for host
  throw sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

template <class WrapperTy, class StdTy, int N>
vector_type_t<StdTy, N>
__esimd_convertvector_from(vector_type_t<element_storage_t<WrapperTy>, N> Val)
#ifdef __SYCL_DEVICE_ONLY__
    ; // needs to be implemented for WrapperTy's for which
      // element_type_traits<WrapperTy>::use_native_cpp_ops is false.
#else
{
  // TODO implement for host
  throw sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

// TODO should be replaced by std::bit_cast once C++20 is supported.
template <class WrapperTy>
WrapperTy __esimd_wrapper_type_bitcast_to(element_storage_t<WrapperTy> Val);
template <class WrapperTy>
element_storage_t<WrapperTy> __esimd_wrapper_type_bitcast_from(WrapperTy Val);

template <class WrapperTy, class StdTy> struct wrapper_type_converter {
  using RawTy = element_storage_t<WrapperTy>;

  template <int N>
  ESIMD_INLINE static vector_type_t<RawTy, N>
  to_vector(vector_type_t<StdTy, N> Val) {
    if constexpr (element_type_traits<WrapperTy>::use_native_cpp_ops) {
      return __builtin_convertvector(Val, vector_type_t<RawTy, N>);
    } else {
      return __esimd_convertvector_to<WrapperTy, StdTy, N>(Val);
    }
  }

  template <int N>
  ESIMD_INLINE static vector_type_t<StdTy, N>
  from_vector(vector_type_t<RawTy, N> Val) {
    if constexpr (element_type_traits<WrapperTy>::use_native_cpp_ops) {
      return __builtin_convertvector(Val, vector_type_t<StdTy, N>);
    } else {
      return __esimd_convertvector_from<WrapperTy, StdTy, N>(Val);
    }
  }
};

// Converts a storage representation of a simd vector with element type
// SrcWrapperTy to a storage representation of a simd vector with element type
// DstWrapperTy.
template <class DstWrapperTy, class SrcWrapperTy, int N,
          class DstRawVecTy = vector_type_t<element_storage_t<DstWrapperTy>, N>,
          class SrcRawVecTy = vector_type_t<element_storage_t<SrcWrapperTy>, N>>
ESIMD_INLINE DstRawVecTy convert_vector(SrcRawVecTy Val) {
  if constexpr (std::is_same_v<SrcWrapperTy, DstWrapperTy>) {
    return Val;
  } else if constexpr (!detail::is_wrapper_elem_type_v<SrcWrapperTy> &&
                       !detail::is_wrapper_elem_type_v<DstWrapperTy>) {
    return __builtin_convertvector(Val, DstRawVecTy);
  } else {
    using DstStdT = typename element_type_traits<DstWrapperTy>::EnclosingCppT;
    using SrcStdT = typename element_type_traits<SrcWrapperTy>::EnclosingCppT;

    using SrcWTC = wrapper_type_converter<SrcWrapperTy, SrcStdT>;
    using DstWTC = wrapper_type_converter<DstWrapperTy, DstStdT>;

    using CommonT = computation_type_t<DstStdT, SrcStdT>;
    using CommonVecT = vector_type_t<CommonT, N>;
    using DstStdVecT = vector_type_t<DstStdT, N>;

    CommonVecT TmpVal;

    // element_storage_t<SrcWrapperTy>
    //                                |
    //                                SrcWrapperTy -- SrcStdT --\
    //                                                          CommonT
    //                                DstWrapperTy -- DstStdT --/
    //                                |
    // element_storage_t<DstWrapperTy>

    if constexpr (std::is_same_v<CommonT, SrcWrapperTy>) {
      TmpVal = std::move(Val);
    } else {
      TmpVal = convert<CommonVecT>(SrcWTC::template from_vector<N>(Val));
    }
    if constexpr (std::is_same_v<DstWrapperTy, CommonT>) {
      return TmpVal;
    } else {
      return DstWTC::template to_vector<N>(convert<DstStdVecT>(TmpVal));
    }
  }
}

template <class Ty>
ESIMD_INLINE element_storage_t<Ty> bitcast_to_storage_type(Ty Val) {
  if constexpr (!is_wrapper_elem_type_v<Ty>) {
    return Val;
  } else {
    return __esimd_wrapper_type_bitcast_from<Ty>(Val);
  }
}

template <class Ty>
ESIMD_INLINE Ty bitcast_to_wrapper_type(element_storage_t<Ty> Val) {
  if constexpr (!is_wrapper_elem_type_v<Ty>) {
    return Val;
  } else {
    return __esimd_wrapper_type_bitcast_to<Ty>(Val);
  }
}

// Converts a scalar value from given source type to destination type. Both
// types can be non-std element types, in which case additional non-C++
// conversions happen if the types are different.
// NOTE: this is not symmetric with convert_vector, which inputs and outputs
// raw (storage) vector types.
template <class DstWrapperTy, class SrcWrapperTy,
          class DstRawTy = element_storage_t<DstWrapperTy>,
          class SrcRawTy = element_storage_t<SrcWrapperTy>>
ESIMD_INLINE DstWrapperTy convert_scalar(SrcWrapperTy Val) {
  if constexpr (std::is_same_v<SrcWrapperTy, DstWrapperTy>) {
    return Val;
  } else if constexpr (!detail::is_wrapper_elem_type_v<SrcWrapperTy> &&
                       !is_wrapper_elem_type_v<DstWrapperTy>) {
    return static_cast<DstRawTy>(Val);
  } else {
    vector_type_t<SrcRawTy, 1> V0 = bitcast_to_storage_type<SrcWrapperTy>(Val);
    vector_type_t<DstRawTy, 1> V1 =
        convert_vector<DstWrapperTy, SrcWrapperTy, 1>(V0);
    return bitcast_to_wrapper_type<DstWrapperTy>(V1[0]);
  }
}

template <BinOp Op, class T> T binary_op_default_impl(T X, T Y) {
  T Res{};
  if constexpr (Op == BinOp::add)
    Res = X + Y;
  else if constexpr (Op == BinOp::sub)
    Res = X - Y;
  else if constexpr (Op == BinOp::mul)
    Res = X * Y;
  else if constexpr (Op == BinOp::div)
    Res = X / Y;
  else if constexpr (Op == BinOp::rem)
    Res = X % Y;
  else if constexpr (Op == BinOp::shl)
    Res = X << Y;
  else if constexpr (Op == BinOp::shr)
    Res = X >> Y;
  else if constexpr (Op == BinOp::bit_or)
    Res = X | Y;
  else if constexpr (Op == BinOp::bit_and)
    Res = X & Y;
  else if constexpr (Op == BinOp::bit_xor)
    Res = X ^ Y;
  else if constexpr (Op == BinOp::log_or)
    Res = X || Y;
  else if constexpr (Op == BinOp::log_and)
    Res = X && Y;
  return Res;
}

template <CmpOp Op, class T> auto comparison_op_default_impl(T X, T Y) {
  decltype(X < Y) Res{};
  if constexpr (Op == CmpOp::lt)
    Res = X < Y;
  else if constexpr (Op == CmpOp::lte)
    Res = X <= Y;
  else if constexpr (Op == CmpOp::eq)
    Res = X == Y;
  else if constexpr (Op == CmpOp::ne)
    Res = X != Y;
  else if constexpr (Op == CmpOp::gte)
    Res = X >= Y;
  else if constexpr (Op == CmpOp::gt)
    Res = X > Y;
  return Res;
}

namespace {
template <class ElemT, int N> struct __hlp {
  using RawElemT = element_storage_t<ElemT>;
  using RawVecT = vector_type_t<RawElemT, N>;
  using BinopT = decltype(std::declval<RawVecT>() + std::declval<RawVecT>());
  using CmpT = decltype(std::declval<RawVecT>() < std::declval<RawVecT>());
};

template <class Hlp> using __re_t = typename Hlp::RawElemT;
template <class Hlp> using __rv_t = typename Hlp::RawVecT;
template <class Hlp> using __cmp_t = typename Hlp::CmpT;
} // namespace

// --- Scalar versions of binary operations

template <BinOp Op, class T> ESIMD_INLINE T __esimd_binary_op(T X, T Y);

template <BinOp Op, class T,
          class = std::enable_if_t<is_valid_simd_elem_type_v<T>>>
ESIMD_INLINE T binary_op_default(T X, T Y) {
  static_assert(element_type_traits<T>::use_native_cpp_ops);
  using T1 = element_storage_t<T>;
  T1 X1 = bitcast_to_storage_type(X);
  T1 Y1 = bitcast_to_storage_type(Y);
  T1 Res = binary_op_default_impl<Op>(X1, Y1);
  return bitcast_to_wrapper_type<T>(Res);
}

template <BinOp Op, class T,
          class = std::enable_if_t<is_valid_simd_elem_type_v<T>>>
ESIMD_INLINE T binary_op(T X, T Y) {
  if constexpr (element_type_traits<T>::use_native_cpp_ops) {
    return binary_op_default<Op>(X, Y);
  } else {
    return __esimd_binary_op<Op>(X, Y);
  }
}

// Default (inefficient) implementation of a scalar binary operation, which
// involves conversion to an std C++ type, performing the op and converting
// back.
template <BinOp Op, class T> ESIMD_INLINE T __esimd_binary_op(T X, T Y) {
  using T1 = typename element_type_traits<T>::EnclosingCppT;
  T1 X1 = convert_scalar<T1, T>(X);
  T1 Y1 = convert_scalar<T1, T>(Y);
  return convert_scalar<T>(binary_op_default<Op, T1>(X1, Y1));
}

// --- Vector versions of binary operations

template <BinOp Op, class ElemT, int N, class RawVecT = __rv_t<__hlp<ElemT, N>>>
ESIMD_INLINE RawVecT vector_binary_op_default(RawVecT X, RawVecT Y) {
  static_assert(element_type_traits<ElemT>::use_native_cpp_ops);
  return binary_op_default_impl<Op, RawVecT>(X, Y);
}

// Default (inefficient) implementation of a vector binary operation, which
// involves conversion to an std C++ type, performing the op and converting
// back.
template <BinOp Op, class ElemT, int N, class RawVecT = __rv_t<__hlp<ElemT, N>>>
ESIMD_INLINE RawVecT __esimd_vector_binary_op(RawVecT X, RawVecT Y) {
  using T1 = element_type_traits<ElemT>::EnclosingCppT;
  using VecT1 = vector_type_t<T1, N>;
  VecT1 X1 = convert_vector<T1, ElemT, N>(X);
  VecT1 Y1 = convert_vector<T1, ElemT, N>(Y);
  return convert_vector<ElemT, T1, N>(
      vector_binary_op_default<Op, T1, N>(X1, Y1));
}

template <BinOp Op, class ElemT, int N, class RawVecT = __rv_t<__hlp<ElemT, N>>>
ESIMD_INLINE RawVecT vector_binary_op(RawVecT X, RawVecT Y) {
  if constexpr (element_type_traits<ElemT>::use_native_cpp_ops) {
    return vector_binary_op_default<Op, ElemT, N>(X, Y);
  } else {
    return __esimd_vector_binary_op<Op, ElemT, N>(X, Y);
  }
}

// --- Vector versions of comparison operations

template <CmpOp Op, class ElemT, int N, class H = __hlp<ElemT, N>,
          class RetT = __cmp_t<H>, class RawVecT = __rv_t<H>>
ESIMD_INLINE RetT vector_comparison_op_default(RawVecT X, RawVecT Y) {
  static_assert(element_type_traits<ElemT>::use_native_cpp_ops);
  return comparison_op_default_impl<Op, RawVecT>(X, Y);
}

// Default (inefficient) implementation of a vector comparison operation, which
// involves conversion to an std C++ type, performing the op and converting
// back.
template <CmpOp Op, class ElemT, int N, class H = __hlp<ElemT, N>,
          class RetT = __cmp_t<H>, class RawVecT = __rv_t<H>>
ESIMD_INLINE RetT __esimd_vector_comparison_op(RawVecT X, RawVecT Y) {
  using T1 = element_type_traits<ElemT>::EnclosingCppT;
  using VecT1 = vector_type_t<T1, N>;
  VecT1 X1 = convert_vector<T1, ElemT, N>(X);
  VecT1 Y1 = convert_vector<T1, ElemT, N>(Y);
  return convert_vector<element_type_t<RetT>, T1>(
      vector_comparison_op_default<Op, T1, N>(X1, Y1));
}

template <CmpOp Op, class ElemT, int N, class H = __hlp<ElemT, N>,
          class RetT = __cmp_t<H>, class RawVecT = __rv_t<H>>
ESIMD_INLINE RetT vector_comparison_op(RawVecT X, RawVecT Y) {
  if constexpr (element_type_traits<ElemT>::use_native_cpp_ops) {
    return vector_comparison_op_default<Op, ElemT, N>(X, Y);
  } else {
    return __esimd_vector_comparison_op<Op, ElemT, N>(X, Y);
  }
}

// Proxy class to access bit representation of a wrapper type both on host and
// device.
// TODO add this functionality to sycl type implementation? With C++20,
// std::bit_cast should be a good replacement.
class WrapperElementTypeProxy {
public:
  template <class T = sycl::half>
  static inline element_storage_t<T> bitcast_from_half(T Val) {
#ifdef __SYCL_DEVICE_ONLY__
    return Val.Data;
#else
    return Val.Data.Buf;
#endif // __SYCL_DEVICE_ONLY__
  }

  template <class T = sycl::half>
  static inline T bitcast_to_half(element_storage_t<T> Bits) {
#ifndef __SYCL_DEVICE_ONLY__
    return sycl::half{Bits};
#else
    sycl::half Res;
    Res.Data = Bits;
    return Res;
#endif // __SYCL_DEVICE_ONLY__
  }
};

// Both std:: variants of the check fail for _Float16, so need a w/a
// template <typename T>
// static inline constexpr bool is_floating_point_v =
// std::is_floating_point_v<element_type_traits<T>::EnclosingCppT>;
//
// template <typename T>
// static inline constexpr bool is_arithmetic_v =
// std::is_arithmetic_v<element_type_traits<T>::EnclosingCppT>;

// @{
// Get computation type of a binary operator given its operand types:
// - if both types are arithmetic - return CPP's "common real type" of the
//   computation (matches C++)
// - if both types are simd types, they must be of the same length N,
//   and the returned type is simd<T, N>, where N is the "common real type" of
//   the element type of the operands (diverges from clang)
// - otherwise, one type is simd and another is arithmetic - the simd type is
//   returned (matches clang)

struct invalid_computation_type;

template <class T1, class T2, class SFINAE = void> struct computation_type {
  using type = invalid_computation_type;
};

template <class T1, class T2>
struct computation_type<T1, T2,
                        std::enable_if_t<is_valid_simd_elem_type_v<T1> &&
                                         is_valid_simd_elem_type_v<T2>>> {
  template <class T> using __tr = element_type_traits<T>;
  template <class T>
  using __native_t = std::conditional_t<__tr<T>::use_native_cpp_ops,
                                        typename __tr<T>::StorageT,
                                        typename __tr<T>::EnclosingCppT>;

  using type =
      decltype(std::declval<__native_t<T1>>() + std::declval<__native_t<T2>>());
};

template <class T1, class T2>
struct computation_type<
    T1, T2,
    std::enable_if_t<is_simd_like_type_v<T1> || is_simd_like_type_v<T2>>> {
private:
  using Ty1 = typename element_type<T1>::type;
  using Ty2 = typename element_type<T2>::type;
  using EltTy = typename computation_type<Ty1, Ty2>::type;
  static constexpr int N1 = is_simd_like_type_v<T1> ? T1::length : 0;
  static constexpr int N2 = is_simd_like_type_v<T2> ? T2::length : 0;
  static_assert((N1 == N2) || ((N1 & N2) == 0), "size mismatch");
  static constexpr int N = N1 ? N1 : N2;

public:
  using type = simd<EltTy, N1>;
};

template <class T1, class T2 = T1>
using computation_type_t =
    typename computation_type<remove_cvref_t<T1>, remove_cvref_t<T2>>::type;

// @}

////////////////////////////////////////////////////////////////////////////////
// sycl::half traits
////////////////////////////////////////////////////////////////////////////////

#ifdef __SYCL_DEVICE_ONLY__
using half = _Float16;
#else
using half = uint16_t;
#endif // __SYCL_DEVICE_ONLY__

template <class T>
struct element_type_traits<T, std::enable_if_t<std::is_same_v<T, sycl::half>>> {
  using EnclosingCppT = float;
  // Can't use sycl::detail::half_impl::StorageT as it still maps to struct on
  // host (even though the struct is a trivial wrapper around uint16_t), and for
  // ESIMD we need a type which can be an element of clang vector.
#ifdef __SYCL_DEVICE_ONLY__
  using StorageT = sycl::detail::half_impl::StorageT;
  // On device, operations on half are translated to operations on _Float16,
  // which is natively supported by the device compiler
  static inline constexpr bool use_native_cpp_ops = true;
#else
  using StorageT = uint16_t;
  // On host, we can't use native Cpp '+', '-' etc. over uint16_t to emulate the
  // operations on half type.
  static inline constexpr bool use_native_cpp_ops = false;
#endif // __SYCL_DEVICE_ONLY__
};

using half_raw = element_storage_t<sycl::half>;

template <>
sycl::half __esimd_wrapper_type_bitcast_to<sycl::half>(half_raw Val) {
  return WrapperElementTypeProxy::bitcast_to_half(Val);
}

template <>
half_raw __esimd_wrapper_type_bitcast_from<sycl::half>(sycl::half Val) {
  return WrapperElementTypeProxy::bitcast_from_half(Val);
}

template <>
struct is_esimd_arithmetic_type<element_storage_t<sycl::half>, void>
    : std::true_type {};

// Misc
inline std::ostream &operator<<(std::ostream &O, half const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

inline std::istream &operator>>(std::istream &I, half &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}

// The only other place which needs to be updated to support a new type is
// the is_wrapper_elem_type_v template constexpr var.

////////////////////////////////////////////////////////////////////////////////
// sycl::bfloat16 traits
////////////////////////////////////////////////////////////////////////////////

} // namespace detail
} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
