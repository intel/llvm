//==------------ - elem_type_traits.hpp - DPC++ Explicit SIMD API ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This header provides basic infrastructure to support non-standard C++ types
// as simd element types. This non-standard element types are usually structs or
// classes (example: sycl::half).
// Terms:
// - "wrapper type" - a non-standard element type
// - "raw type" - the real types used to represent real storage type of the data
//   bits wrapped by the corresponding wrapper structure/class
// By design, user program never uses the raw types, so they are not exposed at
// user level.
//
// The main reasons why the infrastructure is needed are:
// - attempt to create a clang vector with wrapper element type
//   vector_type_t<WrapperT, N> will result in compilation error
// - C++ operations on WrapperT are usually supported by the Intel GPU hardware
//   (which is the main reason of supporting them in ESIMD) and need to be
//   mapped to efficient hardware code sequences.
//
// To make a wrapper type appear as first-class element type, the following
// major components must be available/implemented for the type:
// 1) Storage ("raw") type must be defined. The raw type must be bit-castable to
//   the wrapper type and thus must have the same bit size and alignment
//   requirements.
// 2) "Nearest enclosing" standard C++ type must be defined. This is a standard
//   C++ type which can represent values of the wrapper type. The enclosing type
//   can be used as a fall-back type for default implementations of operations
//   on the wrapper type
// 3) Type conversion intrinsics between the bit representation of a wrapper
//   type value and the equivalent enclosing C++ type value
// 4) The above three are enough to emulate any wrapper type, as all operations
//   can be performed on the enclosing type values, converting from raw to
//   enclosing before the operation and converting back from enclosing to raw
//   after the operation. But this would be inefficient in some cases - when
//   enclosing C++ type does not match the raw type, as H/W usually supports
//   many operations directly on the raw type (which is bit representation of
//   the wrapper type). So mapping to efficient H/W operations must be defined.
//   For example, for SYCL half type efficient mapping primitive operations to
//   Intel GPU harware is as easy as "unwrapping" sycl::half value, which yields
//   "_Float16" natively supported by the device compiler and hardware, then
//   using standard C++, operations such as '+', on _Float16 values. For other
//   types like bfloat16 this will require mapping to appropriate intrinsics.
// 5) The type must be marked as wrapper type explicitly, for the API to behave
//   correctly.
// Important note: some of these components might have different definition for
// the same wrapper type depending on host vs device compilation. E.g. for SYCL
// half the raw type is uint16_t on host and _Float16 on device.
//
// - The mechanism to define components 1) and 2) for a new wrapper type is to
//   provide a specialization of the `element_type_traits` structure for this
//   type.
// - Component 3) is provided via implementing specializations of the following
//   intrinsics:
//   * __esimd_wrapper_type_bitcast_to/__esimd_wrapper_type_bitcast_from (should
//     not be necessary with C++ 20 where there is a standard bitcast operation)
//     to bitcast between the raw and the wrapper types.
//   * __esimd_convertvector_to/__esimd_convertvector_from to type-convert
//     between clang vectors of the wrapper type (bit-represented with the raw
//     type) and clang vectors the the enclosing std type values.
// - Component 4) is provided via:
//   * (primitive operations) Specializations of the
//       __esimd_binary_op
//       __esimd_unary_op
//       __esimd_cmp_op
//       __esimd_vector_binary_op
//       __esimd_vector_unary_op
//       __esimd_vector_cmp_op
//     intrinsics. If the `use_native_cpp_ops` element type trait is true, then
//     implementing those intrinsics is not necessary and std C++ operations
//     will be used.
//   * (math operations) Overloading std math functions for the new wrapper
//     type.
// - Component 5) is provided via adding the new type to the list of types in
//   `is_wrapper_elem_type_v` meta function.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/detail/types.hpp>

#include <CL/sycl/half_type.hpp>

/// @cond ESIMD_DETAIL

__SYCL_INLINE_NAMESPACE(cl) {
namespace __ESIMD_DNS {

// Primitive C++ operations supported by simd objects and templated upon by some
// of the functions/classes.

enum class BinOp {
  add,
  sub,
  mul,
  div,
  rem,
  shl,
  shr,
  bit_or,
  bit_and,
  bit_xor,
  log_or,
  log_and
};

enum class CmpOp { lt, lte, gte, gt, eq, ne };

enum class UnaryOp { minus, plus, bit_not, log_not };

// If given type is a special "wrapper" element type.
template <class T>
static inline constexpr bool is_wrapper_elem_type_v =
    std::is_same_v<T, sycl::half>;

template <class T>
static inline constexpr bool is_valid_simd_elem_type_v =
    (is_vectorizable_v<T> || is_wrapper_elem_type_v<T>);

struct invalid_raw_element_type;

// Default (unusable) definition of the element type traits.
template <class T, class SFINAE> struct element_type_traits {
  // The raw element type of the underlying clang vector used as a
  // storage.
  using RawT = invalid_raw_element_type;
  // A starndard C++ type which this one can be converted to/from.
  // The conversions are usually H/W-supported, and the C++ type can
  // represent the entire range of values of this type.
  using EnclosingCppT = void;
  // Whether a value or clang vector value the raw element type can be used
  // directly as operand to std C++ operations.
  static inline constexpr bool use_native_cpp_ops = true;
  // W/A for MSVC compiler problems which thinks
  // std::is_floating_point_v<_Float16> is false; so require new element types
  // implementations to state "is floating point" trait explicitly
  static inline constexpr bool is_floating_point = false;
};

// Element type traits specialization for C++ standard element type.
template <class T>
struct element_type_traits<T, std::enable_if_t<is_vectorizable_v<T>>> {
  using RawT = T;
  using EnclosingCppT = T;
  static inline constexpr bool use_native_cpp_ops = true;
  static inline constexpr bool is_floating_point = std::is_floating_point_v<T>;
};

// --- Type conversions

// Low-level conversion functions to and from a wrapper element type.
// Must be implemented for each supported
// <wrapper element type, C++ std type pair>.

// These are default implementations for wrapper types with native cpp
// operations support for their corresponding raw type.
template <class WrapperTy, class StdTy, int N>
ESIMD_INLINE vector_type_t<__raw_t<WrapperTy>, N>
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
ESIMD_INLINE vector_type_t<StdTy, N>
__esimd_convertvector_from(vector_type_t<__raw_t<WrapperTy>, N> Val)
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
WrapperTy __esimd_wrapper_type_bitcast_to(__raw_t<WrapperTy> Val);
template <class WrapperTy>
__raw_t<WrapperTy> __esimd_wrapper_type_bitcast_from(WrapperTy Val);

template <class WrapperTy, class StdTy> struct wrapper_type_converter {
  using RawTy = __raw_t<WrapperTy>;

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

// Converts a raw representation of a simd vector with element type
// SrcWrapperTy to a raw representation of a simd vector with element type
// DstWrapperTy.
template <class DstWrapperTy, class SrcWrapperTy, int N,
          class DstRawVecTy = vector_type_t<__raw_t<DstWrapperTy>, N>,
          class SrcRawVecTy = vector_type_t<__raw_t<SrcWrapperTy>, N>>
ESIMD_INLINE DstRawVecTy convert_vector(SrcRawVecTy Val) {
  if constexpr (std::is_same_v<SrcWrapperTy, DstWrapperTy>) {
    return Val;
  } else if constexpr (!is_wrapper_elem_type_v<SrcWrapperTy> &&
                       !is_wrapper_elem_type_v<DstWrapperTy>) {
    return __builtin_convertvector(Val, DstRawVecTy);
  } else {
    // The chain of conversions (some can be no-op if types match):
    // SrcRawVecTy (of SrcWrapperTy)
    //     | step A [wrapper_type_converter<SrcWrapperTy, SrcStdT>]::from_vector
    //     v
    //  SrcStdT
    //     | step B [__builtin_convertvector]
    //     v
    //  DstStdT
    //     | step C [wrapper_type_converter<DstWrapperTy, DstStdT>]::to_vector
    //     v
    // DstRawVecTy (of DstWrapperTy)
    //
    using DstStdT = typename element_type_traits<DstWrapperTy>::EnclosingCppT;
    using SrcStdT = typename element_type_traits<SrcWrapperTy>::EnclosingCppT;
    using SrcConv = wrapper_type_converter<SrcWrapperTy, SrcStdT>;
    using DstConv = wrapper_type_converter<DstWrapperTy, DstStdT>;
    using DstStdVecT = vector_type_t<DstStdT, N>;
    using SrcStdVecT = vector_type_t<SrcStdT, N>;
    SrcStdVecT TmpSrcVal;

    if constexpr (std::is_same_v<SrcStdT, SrcWrapperTy>) {
      TmpSrcVal = std::move(Val);
    } else {
      TmpSrcVal = SrcConv::template from_vector<N>(Val); // step A
    }
    if constexpr (std::is_same_v<SrcStdT, DstWrapperTy>) {
      return TmpSrcVal;
    } else {
      DstStdVecT TmpDstVal;

      if constexpr (std::is_same_v<SrcStdT, DstStdVecT>) {
        TmpDstVal = std::move(TmpSrcVal);
      } else {
        TmpDstVal = __builtin_convertvector(TmpSrcVal, DstStdVecT); // step B
      }
      if constexpr (std::is_same_v<DstStdT, DstWrapperTy>) {
        return TmpDstVal;
      } else {
        return DstConv::template to_vector<N>(TmpDstVal); // step C
      }
    }
  }
}

template <class Ty> ESIMD_INLINE __raw_t<Ty> bitcast_to_raw_type(Ty Val) {
  if constexpr (!is_wrapper_elem_type_v<Ty>) {
    return Val;
  } else {
    return __esimd_wrapper_type_bitcast_from<Ty>(Val);
  }
}

template <class Ty> ESIMD_INLINE Ty bitcast_to_wrapper_type(__raw_t<Ty> Val) {
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
          class DstRawTy = __raw_t<DstWrapperTy>,
          class SrcRawTy = __raw_t<SrcWrapperTy>>
ESIMD_INLINE DstWrapperTy convert_scalar(SrcWrapperTy Val) {
  if constexpr (std::is_same_v<SrcWrapperTy, DstWrapperTy>) {
    return Val;
  } else if constexpr (!is_wrapper_elem_type_v<SrcWrapperTy> &&
                       !is_wrapper_elem_type_v<DstWrapperTy>) {
    return static_cast<DstRawTy>(Val);
  } else {
    vector_type_t<SrcRawTy, 1> V0 = bitcast_to_raw_type<SrcWrapperTy>(Val);
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

template <UnaryOp Op, class T> auto unary_op_default_impl(T X) {
  if constexpr (Op == UnaryOp::minus)
    return -X;
  else if constexpr (Op == UnaryOp::plus)
    return +X;
  else if constexpr (Op == UnaryOp::bit_not)
    return ~X;
  else if constexpr (Op == UnaryOp::log_not)
    return !X;
}

template <class ElemT, int N> struct __hlp {
  using RawElemT = __raw_t<ElemT>;
  using RawVecT = vector_type_t<RawElemT, N>;
  using BinopT = decltype(std::declval<RawVecT>() + std::declval<RawVecT>());
  using CmpT = decltype(std::declval<RawVecT>() < std::declval<RawVecT>());
};

template <class Hlp> using __re_t = typename Hlp::RawElemT;
template <class Hlp> using __rv_t = typename Hlp::RawVecT;
template <class Hlp> using __cmp_t = typename Hlp::CmpT;

// --- Scalar versions of binary operations

template <BinOp Op, class T> ESIMD_INLINE T __esimd_binary_op(T X, T Y);

template <BinOp Op, class T,
          class = std::enable_if_t<is_valid_simd_elem_type_v<T>>>
ESIMD_INLINE T binary_op_default(T X, T Y) {
  static_assert(element_type_traits<T>::use_native_cpp_ops);
  using T1 = __raw_t<T>;
  T1 X1 = bitcast_to_raw_type(X);
  T1 Y1 = bitcast_to_raw_type(Y);
  T1 Res = binary_op_default_impl<Op>(X1, Y1);
  return bitcast_to_wrapper_type<T>(Res);
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

template <BinOp Op, class T,
          class = std::enable_if_t<is_valid_simd_elem_type_v<T>>>
ESIMD_INLINE T binary_op(T X, T Y) {
  if constexpr (element_type_traits<T>::use_native_cpp_ops) {
    return binary_op_default<Op>(X, Y);
  } else {
    return __esimd_binary_op<Op>(X, Y);
  }
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
  using T1 = typename element_type_traits<ElemT>::EnclosingCppT;
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

// --- Scalar versions of unary operations

template <UnaryOp Op, class T> ESIMD_INLINE T __esimd_unary_op(T X);

template <UnaryOp Op, class T,
          class = std::enable_if_t<is_valid_simd_elem_type_v<T>>>
ESIMD_INLINE T unary_op_default(T X) {
  static_assert(element_type_traits<T>::use_native_cpp_ops);
  using T1 = __raw_t<T>;
  T1 X1 = bitcast_to_raw_type(X);
  T1 Res = unary_op_default_impl<Op>(X1);
  return bitcast_to_wrapper_type<T>(Res);
}

// Default (inefficient) implementation of a scalar unary operation, which
// involves conversion to an std C++ type, performing the op and converting
// back.
template <UnaryOp Op, class T> ESIMD_INLINE T __esimd_unary_op(T X) {
  using T1 = typename element_type_traits<T>::EnclosingCppT;
  T1 X1 = convert_scalar<T1, T>(X);
  return convert_scalar<T>(unary_op_default<Op, T1>(X1));
}

template <UnaryOp Op, class T,
          class = std::enable_if_t<is_valid_simd_elem_type_v<T>>>
ESIMD_INLINE T unary_op(T X) {
  if constexpr (element_type_traits<T>::use_native_cpp_ops) {
    return unary_op_default<Op>(X);
  } else {
    return __esimd_unary_op<Op>(X);
  }
}

// --- Vector versions of unary operations

template <UnaryOp Op, class ElemT, int N,
          class RawVecT = __rv_t<__hlp<ElemT, N>>>
ESIMD_INLINE RawVecT vector_unary_op_default(RawVecT X) {
  static_assert(element_type_traits<ElemT>::use_native_cpp_ops);
  return unary_op_default_impl<Op, RawVecT>(X);
}

// Default (inefficient) implementation of a vector unary operation, which
// involves conversion to an std C++ type, performing the op and converting
// back.
template <UnaryOp Op, class ElemT, int N,
          class RawVecT = __rv_t<__hlp<ElemT, N>>>
ESIMD_INLINE RawVecT __esimd_vector_unary_op(RawVecT X) {
  using T1 = typename element_type_traits<ElemT>::EnclosingCppT;
  using VecT1 = vector_type_t<T1, N>;
  VecT1 X1 = convert_vector<T1, ElemT, N>(X);
  return convert_vector<ElemT, T1, N>(vector_unary_op_default<Op, T1, N>(X1));
}

template <UnaryOp Op, class ElemT, int N,
          class RawVecT = __rv_t<__hlp<ElemT, N>>>
ESIMD_INLINE RawVecT vector_unary_op(RawVecT X) {
  if constexpr (element_type_traits<ElemT>::use_native_cpp_ops) {
    return vector_unary_op_default<Op, ElemT, N>(X);
  } else {
    return __esimd_vector_unary_op<Op, ElemT, N>(X);
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
  using T1 = typename element_type_traits<ElemT>::EnclosingCppT;
  using VecT1 = vector_type_t<T1, N>;
  VecT1 X1 = convert_vector<T1, ElemT, N>(X);
  VecT1 Y1 = convert_vector<T1, ElemT, N>(Y);
  return convert_vector<element_type_t<RetT>, T1, N>(
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
  static inline __raw_t<T> bitcast_from_half(T Val) {
#ifdef __SYCL_DEVICE_ONLY__
    return Val.Data;
#else
    return Val.Data.Buf;
#endif // __SYCL_DEVICE_ONLY__
  }

  template <class T = sycl::half>
  static inline T bitcast_to_half(__raw_t<T> Bits) {
#ifndef __SYCL_DEVICE_ONLY__
    return sycl::half{Bits};
#else
    sycl::half Res;
    Res.Data = Bits;
    return Res;
#endif // __SYCL_DEVICE_ONLY__
  }
};

// "Generic" version of std::is_floating_point_v which returns "true" also for
// the wrapper floating-point types such as sycl::half.
template <typename T>
static inline constexpr bool is_generic_floating_point_v =
    element_type_traits<T>::is_floating_point;

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
private:
  template <class T> using tr = element_type_traits<T>;
  template <class T>
  using native_t =
      std::conditional_t<tr<T>::use_native_cpp_ops, typename tr<T>::RawT,
                         typename tr<T>::EnclosingCppT>;
  static inline constexpr bool is_wr1 = is_wrapper_elem_type_v<T1>;
  static inline constexpr bool is_wr2 = is_wrapper_elem_type_v<T2>;
  static inline constexpr bool is_fp1 = is_generic_floating_point_v<T1>;
  static inline constexpr bool is_fp2 = is_generic_floating_point_v<T2>;

public:
  using type = std::conditional_t<
      !is_wr1 && !is_wr2,
      // T1 and T2 are both std C++ types - use std C++ type promotion
      decltype(std::declval<T1>() + std::declval<T2>()),
      std::conditional_t<
          std::is_same_v<T1, T2>,
          // Types are the same wrapper type - return any
          T1,
          std::conditional_t<is_fp1 != is_fp2,
                             // One of the types is floating-point - return it
                             // (e.g. computation_type<int, sycl::half> will
                             // yield sycl::half)
                             std::conditional_t<is_fp1, T1, T2>,
                             // both are either floating point or integral -
                             // return result of C++ promotion of the native
                             // types
                             decltype(std::declval<native_t<T1>>() +
                                      std::declval<native_t<T2>>())>>>;
};

template <class T1, class T2>
struct computation_type<
    T1, T2,
    std::enable_if_t<is_simd_like_type_v<T1> || is_simd_like_type_v<T2>>> {
private:
  using Ty1 = element_type_t<T1>;
  using Ty2 = element_type_t<T2>;
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

////////////////////////////////////////////////////////////////////////////////
// sycl::half traits
////////////////////////////////////////////////////////////////////////////////

template <class T>
struct element_type_traits<T, std::enable_if_t<std::is_same_v<T, sycl::half>>> {
  // Can't use sycl::detail::half_impl::StorageT as RawT for both host and
  // device as it still maps to struct on/ host (even though the struct is a
  // trivial wrapper around uint16_t), and for ESIMD we need a type which can be
  // an element of clang vector.
#ifdef __SYCL_DEVICE_ONLY__
  using RawT = sycl::detail::half_impl::StorageT;
  // On device, _Float16 is native Cpp type, so it is the enclosing C++ type
  using EnclosingCppT = RawT;
  // On device, operations on half are translated to operations on _Float16,
  // which is natively supported by the device compiler
  static inline constexpr bool use_native_cpp_ops = true;
#else
  using RawT = uint16_t;
  using EnclosingCppT = float;
  // On host, we can't use native Cpp '+', '-' etc. over uint16_t to emulate the
  // operations on half type.
  static inline constexpr bool use_native_cpp_ops = false;
#endif // __SYCL_DEVICE_ONLY__

  static inline constexpr bool is_floating_point = true;
};

using half_raw = __raw_t<sycl::half>;

template <>
ESIMD_INLINE sycl::half
__esimd_wrapper_type_bitcast_to<sycl::half>(half_raw Val) {
  return WrapperElementTypeProxy::bitcast_to_half(Val);
}

template <>
ESIMD_INLINE half_raw
__esimd_wrapper_type_bitcast_from<sycl::half>(sycl::half Val) {
  return WrapperElementTypeProxy::bitcast_from_half(Val);
}

template <>
struct is_esimd_arithmetic_type<__raw_t<sycl::half>, void> : std::true_type {};

// Misc
inline std::ostream &operator<<(std::ostream &O, sycl::half const &rhs) {
  O << static_cast<float>(rhs);
  return O;
}

inline std::istream &operator>>(std::istream &I, sycl::half &rhs) {
  float ValFloat = 0.0f;
  I >> ValFloat;
  rhs = ValFloat;
  return I;
}

// The only other place which needs to be updated to support a new type is
// the is_wrapper_elem_type_v meta function.

////////////////////////////////////////////////////////////////////////////////
// sycl::bfloat16 traits
////////////////////////////////////////////////////////////////////////////////
// TODO

} // namespace __ESIMD_DNS
} // __SYCL_INLINE_NAMESPACE(cl)

/// @endcond ESIMD_DETAIL
