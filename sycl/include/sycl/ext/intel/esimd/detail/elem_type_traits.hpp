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
// Important note: some of these components might have different definition for
// the same wrapper type depending on host vs device compilation. E.g. for SYCL
// half the raw type is uint16_t on host and _Float16 on device.
//
// - The mechanism to define components 1) and 2) for a new wrapper type is to
//   provide a specialization of the `element_type_traits` structure for this
//   type.
// - Component 3) is provided via implementing specializations of the
//   conversion traits:
//   * scalar_conversion_traits: functions to bitcast between the raw and the
//     wrapper types (should not be necessary with C++ 20 where there is a
//     standard bitcast operation)
//   * vector_conversion_traits: functions to type-convert between clang
//     vectors of the wrapper type (bit-represented with the raw type) and clang
//     vectors the the enclosing std type values.
// - Component 4) is provided via:
//   * (primitive operations) Specializations of the
//     - scalar_binary_op_traits
//     - vector_binary_op_traits
//     - scalar_unary_op_traits
//     - vector_unary_op_traits
//     - scalar_comparison_op_traits
//     - vector_comparison_op_traits
//     structs. If the `use_native_cpp_ops` element type trait is true, then
//     implementing those specializations is not necessary and std C++
//     operations will be used.
//   * (math operations) Overloading std math functions for the new wrapper
//     type.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/common.hpp>
#include <sycl/ext/intel/esimd/detail/defines_elementary.hpp>
#include <sycl/ext/intel/esimd/detail/types_elementary.hpp>

#include <utility>

/// @cond ESIMD_DETAIL

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd::detail {

// -----------------------------------------------------------------------------
// General declarations
// -----------------------------------------------------------------------------

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

struct invalid_raw_element_type;

// -----------------------------------------------------------------------------
// Traits to be implemented for wrapper types (interleaving with some useful
// meta-functions and declarations).
// -----------------------------------------------------------------------------

// ------------------- Basic type traits

// Default (unusable) definition of the element type traits.
template <class T, class SFINAE = void> struct element_type_traits {
  // The raw element type of the underlying clang vector used as a
  // storage.
  using RawT = invalid_raw_element_type;
  // A starndard C++ type which this one can be converted to/from.
  // The conversions are usually H/W-supported, and the C++ type can
  // represent the entire range of values of this type.
  using EnclosingCppT = void;
  // Whether a value or clang vector value the raw element type can be used
  // directly as operand to std C++ operations.
  static constexpr bool use_native_cpp_ops = true;
  // W/A for MSVC compiler problems which thinks
  // std::is_floating_point_v<_Float16> is false; so require new element types
  // implementations to state "is floating point" trait explicitly
  static constexpr bool is_floating_point = false;
};

// Element type traits specialization for C++ standard element type.
template <class T>
struct element_type_traits<T, std::enable_if_t<is_vectorizable_v<T>>> {
  using RawT = T;
  using EnclosingCppT = T;
  static constexpr bool use_native_cpp_ops = true;
  static constexpr bool is_floating_point = std::is_floating_point_v<T>;
};

// ------------------- Useful meta-functions and declarations

template <class T> using __raw_t = typename element_type_traits<T>::RawT;
template <class T>
using __cpp_t = typename element_type_traits<T>::EnclosingCppT;

template <class T, int N>
using __raw_vec_t = vector_type_t<typename element_type_traits<T>::RawT, N>;

// Note: using RawVecT in comparison result type calculation does *not* mean
// the comparison is actually performed on the raw types.
template <class T, int N>
using __cmp_t = decltype(std::declval<__raw_vec_t<T, N>>() <
                         std::declval<__raw_vec_t<T, N>>());

// Is given type is a special "wrapper" element type?
template <class T>
static inline constexpr bool is_wrapper_elem_type_v =
    !std::is_same_v<__raw_t<T>, invalid_raw_element_type> &&
    !std::is_same_v<__raw_t<T>, T>;

template <class T>
static inline constexpr bool is_valid_simd_elem_type_v =
    (is_vectorizable_v<T> || is_wrapper_elem_type_v<T>);

// ------------------- Type conversion traits

template <class WrapperT, int N> struct vector_conversion_traits {
  static_assert(is_wrapper_elem_type_v<WrapperT>, "");
  using StdT = __cpp_t<WrapperT>;
  using RawT = __raw_t<WrapperT>;

  static vector_type_t<RawT, N> convert_to_raw(vector_type_t<StdT, N>);
  static vector_type_t<StdT, N> convert_to_cpp(vector_type_t<RawT, N>);
};

template <class WrapperT> struct scalar_conversion_traits {
  static_assert(is_wrapper_elem_type_v<WrapperT>, "");
  using RawT = __raw_t<WrapperT>;

  static RawT bitcast_to_raw(WrapperT);
  static WrapperT bitcast_to_wrapper(RawT);
};

// ------------------- Binary operation traits

template <BinOp Op, class WrapperT> struct scalar_binary_op_traits {
  static_assert(is_wrapper_elem_type_v<WrapperT>, "");

  static WrapperT impl(WrapperT X, WrapperT Y);
};

template <BinOp Op, class WrapperT, int N> struct vector_binary_op_traits {
  static_assert(is_wrapper_elem_type_v<WrapperT>, "");
  using RawVecT = __raw_vec_t<WrapperT, N>;

  static RawVecT impl(RawVecT X, RawVecT Y);
};

// ------------------- Comparison operation traits

template <CmpOp Op, class WrapperT> struct scalar_comparison_op_traits {
  static_assert(is_wrapper_elem_type_v<WrapperT>, "");

  static bool impl(WrapperT X, WrapperT Y);
};

template <CmpOp Op, class WrapperT, int N> struct vector_comparison_op_traits {
  static_assert(is_wrapper_elem_type_v<WrapperT>, "");
  using RawVecT = __raw_vec_t<WrapperT, N>;

  static __cmp_t<WrapperT, N> impl(RawVecT X, RawVecT Y);
};

// ------------------- Unary operation traits

template <UnaryOp Op, class WrapperT> struct scalar_unary_op_traits {
  static_assert(is_wrapper_elem_type_v<WrapperT>, "");

  static WrapperT impl(WrapperT X);
};

template <UnaryOp Op, class WrapperT, int N> struct vector_unary_op_traits {
  static_assert(is_wrapper_elem_type_v<WrapperT>, "");
  using RawVecT = __raw_vec_t<WrapperT, N>;

  static RawVecT impl(RawVecT X);
};

// -----------------------------------------------------------------------------
// Main type conversion meta-functions used in traits implementations and other
// ESIMD components.
// -----------------------------------------------------------------------------

template <class WrapperT> struct wrapper_type_converter {
  using StdT = __cpp_t<WrapperT>;
  using RawT = __raw_t<WrapperT>;

  template <int N>
  ESIMD_INLINE static vector_type_t<RawT, N>
  to_vector(vector_type_t<StdT, N> Val) {
    if constexpr (element_type_traits<WrapperT>::use_native_cpp_ops) {
      return __builtin_convertvector(Val, vector_type_t<RawT, N>);
    } else {
      return vector_conversion_traits<WrapperT, N>::convert_to_raw(Val);
    }
  }

  template <int N>
  ESIMD_INLINE static vector_type_t<StdT, N>
  from_vector(vector_type_t<RawT, N> Val) {
    if constexpr (element_type_traits<WrapperT>::use_native_cpp_ops) {
      return __builtin_convertvector(Val, vector_type_t<StdT, N>);
    } else {
      return vector_conversion_traits<WrapperT, N>::convert_to_cpp(Val);
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
    //     | step A [wrapper_type_converter<SrcWrapperTy>]::from_vector
    //     v
    //  SrcStdT
    //     | step B [__builtin_convertvector]
    //     v
    //  DstStdT
    //     | step C [wrapper_type_converter<DstWrapperTy>]::to_vector
    //     v
    // DstRawVecTy (of DstWrapperTy)
    //
    using SrcConv = wrapper_type_converter<SrcWrapperTy>;
    using DstConv = wrapper_type_converter<DstWrapperTy>;
    using SrcStdT = typename SrcConv::StdT;
    using DstStdT = typename DstConv::StdT;
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

// -----------------------------------------------------------------------------
// Implementations of standard C++ operations - (comparison, binary and unary)
// for the vectors and scalars of wrapper types based the traits declared above.
// -----------------------------------------------------------------------------

template <class Ty> ESIMD_INLINE __raw_t<Ty> bitcast_to_raw_type(Ty Val) {
  if constexpr (!is_wrapper_elem_type_v<Ty>) {
    return Val;
  } else {
    return scalar_conversion_traits<Ty>::bitcast_to_raw(Val);
  }
}

template <class Ty> ESIMD_INLINE Ty bitcast_to_wrapper_type(__raw_t<Ty> Val) {
  if constexpr (!is_wrapper_elem_type_v<Ty>) {
    return Val;
  } else {
    return scalar_conversion_traits<Ty>::bitcast_to_wrapper(Val);
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

// Default implementation of a binary arithmetic operation. Works for both
// scalar and vector types.
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

// Default implementation of a comparison operation. Works for both scalar and
// vector types.
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

// Default implementation of an unary operation. Works for both scalar and
// vector types.
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

// --- Scalar versions of binary operations

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

template <BinOp Op, class T,
          class = std::enable_if_t<is_valid_simd_elem_type_v<T>>>
ESIMD_INLINE T binary_op(T X, T Y) {
  if constexpr (element_type_traits<T>::use_native_cpp_ops) {
    return binary_op_default<Op>(X, Y);
  } else {
    return scalar_binary_op_traits<Op, T>::impl(X, Y);
  }
}

// --- Vector versions of binary operations

template <BinOp Op, class ElemT, int N, class RawVecT = __raw_vec_t<ElemT, N>>
ESIMD_INLINE RawVecT vector_binary_op_default(RawVecT X, RawVecT Y) {
  static_assert(element_type_traits<ElemT>::use_native_cpp_ops);
  return binary_op_default_impl<Op, RawVecT>(X, Y);
}

template <BinOp Op, class ElemT, int N, class RawVecT = __raw_vec_t<ElemT, N>>
ESIMD_INLINE RawVecT vector_binary_op(RawVecT X, RawVecT Y) {
  if constexpr (element_type_traits<ElemT>::use_native_cpp_ops) {
    return vector_binary_op_default<Op, ElemT, N>(X, Y);
  } else {
    return vector_binary_op_traits<Op, ElemT, N>::impl(X, Y);
  }
}

// --- Scalar versions of unary operations

template <UnaryOp Op, class T,
          class = std::enable_if_t<is_valid_simd_elem_type_v<T>>>
ESIMD_INLINE T unary_op_default(T X) {
  static_assert(element_type_traits<T>::use_native_cpp_ops);
  using T1 = __raw_t<T>;
  T1 X1 = bitcast_to_raw_type(X);
  T1 Res = unary_op_default_impl<Op>(X1);
  return bitcast_to_wrapper_type<T>(Res);
}

template <UnaryOp Op, class T,
          class = std::enable_if_t<is_valid_simd_elem_type_v<T>>>
ESIMD_INLINE T unary_op(T X) {
  if constexpr (element_type_traits<T>::use_native_cpp_ops) {
    return unary_op_default<Op>(X);
  } else {
    return scalar_unary_op_traits<Op, T>::impl(X);
  }
}

// --- Vector versions of unary operations

template <UnaryOp Op, class ElemT, int N, class RawVecT = __raw_vec_t<ElemT, N>>
ESIMD_INLINE RawVecT vector_unary_op_default(RawVecT X) {
  static_assert(element_type_traits<ElemT>::use_native_cpp_ops);
  return unary_op_default_impl<Op, RawVecT>(X);
}

template <UnaryOp Op, class ElemT, int N, class RawVecT = __raw_vec_t<ElemT, N>>
ESIMD_INLINE RawVecT vector_unary_op(RawVecT X) {
  if constexpr (element_type_traits<ElemT>::use_native_cpp_ops) {
    return vector_unary_op_default<Op, ElemT, N>(X);
  } else {
    return vector_unary_op_traits<Op, ElemT, N>::impl(X);
  }
}

// --- Vector versions of comparison operations

template <CmpOp Op, class ElemT, int N, class RetT = __cmp_t<ElemT, N>,
          class RawVecT = __raw_vec_t<ElemT, N>>
ESIMD_INLINE RetT vector_comparison_op_default(RawVecT X, RawVecT Y) {
  static_assert(element_type_traits<ElemT>::use_native_cpp_ops);
  return comparison_op_default_impl<Op, RawVecT>(X, Y);
}

template <CmpOp Op, class ElemT, int N, class RetT = __cmp_t<ElemT, N>,
          class RawVecT = __raw_vec_t<ElemT, N>>
ESIMD_INLINE RetT vector_comparison_op(RawVecT X, RawVecT Y) {
  if constexpr (element_type_traits<ElemT>::use_native_cpp_ops) {
    return vector_comparison_op_default<Op, ElemT, N>(X, Y);
  } else {
    return vector_comparison_op_traits<Op, ElemT, N>::impl(X, Y);
  }
}

// -----------------------------------------------------------------------------
// Default implementations of the traits (used in the operations above).
// -----------------------------------------------------------------------------

// Default (inefficient) implementation of a scalar binary operation, which
// involves conversion to an std C++ type, performing the op and converting
// back.
template <BinOp Op, class WrapperT>
ESIMD_INLINE WrapperT scalar_binary_op_traits<Op, WrapperT>::impl(WrapperT X,
                                                                  WrapperT Y) {
  using T1 = __cpp_t<WrapperT>;
  T1 X1 = convert_scalar<T1, WrapperT>(X);
  T1 Y1 = convert_scalar<T1, WrapperT>(Y);
  return convert_scalar<WrapperT>(binary_op_default<Op, T1>(X1, Y1));
}

// Default (inefficient) implementation of a vector binary operation, which
// involves conversion to an std C++ type, performing the op and converting
// back.
template <BinOp Op, class WrapperT, int N>
ESIMD_INLINE __raw_vec_t<WrapperT, N>
vector_binary_op_traits<Op, WrapperT, N>::impl(__raw_vec_t<WrapperT, N> X,
                                               __raw_vec_t<WrapperT, N> Y) {
  using T1 = __cpp_t<WrapperT>;
  using VecT1 = vector_type_t<T1, N>;
  VecT1 X1 = convert_vector<T1, WrapperT, N>(X);
  VecT1 Y1 = convert_vector<T1, WrapperT, N>(Y);
  return convert_vector<WrapperT, T1, N>(
      vector_binary_op_default<Op, T1, N>(X1, Y1));
}

// Default (inefficient) implementation of a scalar unary operation, which
// involves conversion to an std C++ type, performing the op and converting
// back.
template <UnaryOp Op, class WrapperT>
ESIMD_INLINE WrapperT scalar_unary_op_traits<Op, WrapperT>::impl(WrapperT X) {
  using T1 = __cpp_t<WrapperT>;
  T1 X1 = convert_scalar<T1, WrapperT>(X);
  return convert_scalar<WrapperT>(unary_op_default<Op, T1>(X1));
}

// Default (inefficient) implementation of a vector unary operation, which
// involves conversion to an std C++ type, performing the op and converting
// back.
template <UnaryOp Op, class WrapperT, int N>
ESIMD_INLINE __raw_vec_t<WrapperT, N>
vector_unary_op_traits<Op, WrapperT, N>::impl(__raw_vec_t<WrapperT, N> X) {
  using T1 = __cpp_t<WrapperT>;
  using VecT1 = vector_type_t<T1, N>;
  VecT1 X1 = convert_vector<T1, WrapperT, N>(X);
  return convert_vector<WrapperT, T1, N>(
      vector_unary_op_default<Op, T1, N>(X1));
}

// Default (inefficient) implementation of a vector comparison operation, which
// involves conversion to an std C++ type, performing the op and converting
// back.
template <CmpOp Op, class WrapperT, int N>
ESIMD_INLINE __cmp_t<WrapperT, N>
vector_comparison_op_traits<Op, WrapperT, N>::impl(__raw_vec_t<WrapperT, N> X,
                                                   __raw_vec_t<WrapperT, N> Y) {
  using T1 = __cpp_t<WrapperT>;
  using VecT1 = vector_type_t<T1, N>;
  VecT1 X1 = convert_vector<T1, WrapperT, N>(X);
  VecT1 Y1 = convert_vector<T1, WrapperT, N>(Y);
  return convert_vector<vector_element_type_t<__cmp_t<WrapperT, N>>, T1, N>(
      vector_comparison_op_default<Op, T1, N>(X1, Y1));
}

// "Generic" version of std::is_floating_point_v which returns "true" also for
// the wrapper floating-point types such as sycl::half.
template <typename T>
static inline constexpr bool is_generic_floating_point_v =
    element_type_traits<T>::is_floating_point;

} // namespace ext::intel::esimd::detail
} // namespace _V1
} // namespace sycl

/// @endcond ESIMD_DETAIL
