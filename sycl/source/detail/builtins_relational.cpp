//==------ builtins_relational.cpp - SYCL built-in relational functions ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines the host versions of functions defined
// in SYCL SPEC section - 4.13.7 Relational functions.

#include "builtins_helper.hpp"
#include <CL/sycl/detail/stl_type_traits.hpp>

#include <cmath>

namespace s = cl::sycl;
namespace d = s::detail;

__SYCL_INLINE_NAMESPACE(cl) {
namespace __host_std {
namespace {

template <typename T> inline T __vFOrdEqual(T x, T y) { return -(x == y); }

template <typename T> inline T __sFOrdEqual(T x, T y) { return x == y; }

template <typename T> inline T __vFUnordNotEqual(T x, T y) { return -(x != y); }

template <typename T> inline T __sFUnordNotEqual(T x, T y) { return x != y; }

template <typename T> inline T __vFOrdGreaterThan(T x, T y) { return -(x > y); }

template <typename T> inline T __sFOrdGreaterThan(T x, T y) { return x > y; }

template <typename T> inline T __vFOrdGreaterThanEqual(T x, T y) {
  return -(x >= y);
}

template <typename T> inline T __sFOrdGreaterThanEqual(T x, T y) {
  return x >= y;
}

template <typename T> inline T __vFOrdLessThanEqual(T x, T y) {
  return -(x <= y);
}

template <typename T> inline T __sFOrdLessThanEqual(T x, T y) { return x <= y; }

template <typename T> inline T __vLessOrGreater(T x, T y) {
  return -((x < y) || (x > y));
}

template <typename T> inline T __sLessOrGreater(T x, T y) {
  return ((x < y) || (x > y));
}

template <typename T> s::cl_int inline __Any(T x) { return d::msbIsSet(x); }
template <typename T> s::cl_int inline __All(T x) { return d::msbIsSet(x); }

template <typename T> inline T __vOrdered(T x, T y) {
  return -(
      !(std::isunordered(d::cast_if_host_half(x), d::cast_if_host_half(y))));
}

template <typename T> inline T __sOrdered(T x, T y) {
  return !(std::isunordered(d::cast_if_host_half(x), d::cast_if_host_half(y)));
}

template <typename T> inline T __vUnordered(T x, T y) {
  return -(static_cast<T>(
      std::isunordered(d::cast_if_host_half(x), d::cast_if_host_half(y))));
}

template <typename T> inline T __sUnordered(T x, T y) {
  return std::isunordered(d::cast_if_host_half(x), d::cast_if_host_half(y));
}

template <typename T>
inline typename sycl::detail::enable_if_t<d::is_sgeninteger<T>::value, T>
__bitselect(T a, T b, T c) {
  return (a & ~c) | (b & c);
}

template <typename T> union databitset;
// cl_float
template <> union databitset<s::cl_float> {
  static_assert(sizeof(s::cl_int) == sizeof(s::cl_float),
                "size of cl_float is not equal to 32 bits(cl_int).");
  s::cl_float f;
  s::cl_int i;
};

// cl_double
template <> union databitset<s::cl_double> {
  static_assert(sizeof(s::cl_long) == sizeof(s::cl_double),
                "size of cl_double is not equal to 64 bits(cl_long).");
  s::cl_double f;
  s::cl_long i;
};

// cl_half
template <> union databitset<s::cl_half> {
  static_assert(sizeof(s::cl_short) == sizeof(s::cl_half),
                "size of cl_half is not equal to 16 bits(cl_short).");
  s::cl_half f;
  s::cl_short i;
};

template <typename T>
typename sycl::detail::enable_if_t<d::is_sgenfloat<T>::value,
                                   T> inline __bitselect(T a, T b, T c) {
  databitset<T> ba;
  ba.f = a;
  databitset<T> bb;
  bb.f = b;
  databitset<T> bc;
  bc.f = c;
  databitset<T> br;
  br.f = 0;
  br.i = ((ba.i & ~bc.i) | (bb.i & bc.i));
  return br.f;
}

template <typename T, typename T2> inline T2 __select(T2 a, T2 b, T c) {
  return (c ? b : a);
}

template <typename T, typename T2> inline T2 __vselect(T2 a, T2 b, T c) {
  return d::msbIsSet(c) ? b : a;
}
} // namespace

// ---------- 4.13.7 Relational functions. Host implementations. ---------------
// FOrdEqual-isequal
__SYCL_EXPORT s::cl_int FOrdEqual(s::cl_float x, s::cl_float y) __NOEXC {
  return __sFOrdEqual(x, y);
}
__SYCL_EXPORT s::cl_int FOrdEqual(s::cl_double x, s::cl_double y) __NOEXC {
  return __sFOrdEqual(x, y);
}
__SYCL_EXPORT s::cl_int FOrdEqual(s::cl_half x, s::cl_half y) __NOEXC {
  return __sFOrdEqual(x, y);
}
MAKE_1V_2V_FUNC(FOrdEqual, __vFOrdEqual, s::cl_int, s::cl_float, s::cl_float)
MAKE_1V_2V_FUNC(FOrdEqual, __vFOrdEqual, s::cl_long, s::cl_double, s::cl_double)
MAKE_1V_2V_FUNC(FOrdEqual, __vFOrdEqual, s::cl_short, s::cl_half, s::cl_half)

// FUnordNotEqual-isnotequal
__SYCL_EXPORT s::cl_int FUnordNotEqual(s::cl_float x, s::cl_float y) __NOEXC {
  return __sFUnordNotEqual(x, y);
}
__SYCL_EXPORT s::cl_int FUnordNotEqual(s::cl_double x, s::cl_double y) __NOEXC {
  return __sFUnordNotEqual(x, y);
}
__SYCL_EXPORT s::cl_int FUnordNotEqual(s::cl_half x, s::cl_half y) __NOEXC {
  return __sFUnordNotEqual(x, y);
}
MAKE_1V_2V_FUNC(FUnordNotEqual, __vFUnordNotEqual, s::cl_int, s::cl_float,
                s::cl_float)
MAKE_1V_2V_FUNC(FUnordNotEqual, __vFUnordNotEqual, s::cl_long, s::cl_double,
                s::cl_double)
MAKE_1V_2V_FUNC(FUnordNotEqual, __vFUnordNotEqual, s::cl_short, s::cl_half,
                s::cl_half)

// (FOrdGreaterThan)      // isgreater
__SYCL_EXPORT s::cl_int FOrdGreaterThan(s::cl_float x, s::cl_float y) __NOEXC {
  return __sFOrdGreaterThan(x, y);
}
__SYCL_EXPORT s::cl_int FOrdGreaterThan(s::cl_double x,
                                        s::cl_double y) __NOEXC {
  return __sFOrdGreaterThan(x, y);
}
__SYCL_EXPORT s::cl_int FOrdGreaterThan(s::cl_half x, s::cl_half y) __NOEXC {
  return __sFOrdGreaterThan(x, y);
}
MAKE_1V_2V_FUNC(FOrdGreaterThan, __vFOrdGreaterThan, s::cl_int, s::cl_float,
                s::cl_float)
MAKE_1V_2V_FUNC(FOrdGreaterThan, __vFOrdGreaterThan, s::cl_long, s::cl_double,
                s::cl_double)
MAKE_1V_2V_FUNC(FOrdGreaterThan, __vFOrdGreaterThan, s::cl_short, s::cl_half,
                s::cl_half)

// (FOrdGreaterThanEqual) // isgreaterequal
__SYCL_EXPORT s::cl_int FOrdGreaterThanEqual(s::cl_float x,
                                             s::cl_float y) __NOEXC {
  return __sFOrdGreaterThanEqual(x, y);
}
__SYCL_EXPORT s::cl_int FOrdGreaterThanEqual(s::cl_double x,
                                             s::cl_double y) __NOEXC {
  return __sFOrdGreaterThanEqual(x, y);
}
__SYCL_EXPORT s::cl_int FOrdGreaterThanEqual(s::cl_half x,
                                             s::cl_half y) __NOEXC {
  return __sFOrdGreaterThanEqual(x, y);
}
MAKE_1V_2V_FUNC(FOrdGreaterThanEqual, __vFOrdGreaterThanEqual, s::cl_int,
                s::cl_float, s::cl_float)
MAKE_1V_2V_FUNC(FOrdGreaterThanEqual, __vFOrdGreaterThanEqual, s::cl_long,
                s::cl_double, s::cl_double)
MAKE_1V_2V_FUNC(FOrdGreaterThanEqual, __vFOrdGreaterThanEqual, s::cl_short,
                s::cl_half, s::cl_half)

// (FOrdLessThan)         // isless
__SYCL_EXPORT s::cl_int FOrdLessThan(s::cl_float x, s::cl_float y) __NOEXC {
  return (x < y);
}
__SYCL_EXPORT s::cl_int FOrdLessThan(s::cl_double x, s::cl_double y) __NOEXC {
  return (x < y);
}
__SYCL_EXPORT s::cl_int __vFOrdLessThan(s::cl_float x, s::cl_float y) __NOEXC {
  return -(x < y);
}
__SYCL_EXPORT s::cl_long __vFOrdLessThan(s::cl_double x,
                                         s::cl_double y) __NOEXC {
  return -(x < y);
}
__SYCL_EXPORT s::cl_int FOrdLessThan(s::cl_half x, s::cl_half y) __NOEXC {
  return (x < y);
}
__SYCL_EXPORT s::cl_short __vFOrdLessThan(s::cl_half x, s::cl_half y) __NOEXC {
  return -(x < y);
}
MAKE_1V_2V_FUNC(FOrdLessThan, __vFOrdLessThan, s::cl_int, s::cl_float,
                s::cl_float)
MAKE_1V_2V_FUNC(FOrdLessThan, __vFOrdLessThan, s::cl_long, s::cl_double,
                s::cl_double)
MAKE_1V_2V_FUNC(FOrdLessThan, __vFOrdLessThan, s::cl_short, s::cl_half,
                s::cl_half)

// (FOrdLessThanEqual)    // islessequal
__SYCL_EXPORT s::cl_int FOrdLessThanEqual(s::cl_float x,
                                          s::cl_float y) __NOEXC {
  return __sFOrdLessThanEqual(x, y);
}
__SYCL_EXPORT s::cl_int FOrdLessThanEqual(s::cl_double x,
                                          s::cl_double y) __NOEXC {
  return __sFOrdLessThanEqual(x, y);
}
__SYCL_EXPORT s::cl_int FOrdLessThanEqual(s::cl_half x, s::cl_half y) __NOEXC {
  return __sFOrdLessThanEqual(x, y);
}
MAKE_1V_2V_FUNC(FOrdLessThanEqual, __vFOrdLessThanEqual, s::cl_int, s::cl_float,
                s::cl_float)
MAKE_1V_2V_FUNC(FOrdLessThanEqual, __vFOrdLessThanEqual, s::cl_long,
                s::cl_double, s::cl_double)
MAKE_1V_2V_FUNC(FOrdLessThanEqual, __vFOrdLessThanEqual, s::cl_short,
                s::cl_half, s::cl_half)

// (LessOrGreater)        // islessgreater
__SYCL_EXPORT s::cl_int LessOrGreater(s::cl_float x, s::cl_float y) __NOEXC {
  return __sLessOrGreater(x, y);
}
__SYCL_EXPORT s::cl_int LessOrGreater(s::cl_double x, s::cl_double y) __NOEXC {
  return __sLessOrGreater(x, y);
}
__SYCL_EXPORT s::cl_int LessOrGreater(s::cl_half x, s::cl_half y) __NOEXC {
  return __sLessOrGreater(x, y);
}
MAKE_1V_2V_FUNC(LessOrGreater, __vLessOrGreater, s::cl_int, s::cl_float,
                s::cl_float)
MAKE_1V_2V_FUNC(LessOrGreater, __vLessOrGreater, s::cl_long, s::cl_double,
                s::cl_double)
MAKE_1V_2V_FUNC(LessOrGreater, __vLessOrGreater, s::cl_short, s::cl_half,
                s::cl_half)

// (IsFinite)             // isfinite
__SYCL_EXPORT s::cl_int IsFinite(s::cl_float x) __NOEXC {
  return std::isfinite(x);
}
__SYCL_EXPORT s::cl_int IsFinite(s::cl_double x) __NOEXC {
  return std::isfinite(x);
}
__SYCL_EXPORT s::cl_int __vIsFinite(s::cl_float x) __NOEXC {
  return -static_cast<s::cl_int>(std::isfinite(x));
}
__SYCL_EXPORT s::cl_long __vIsFinite(s::cl_double x) __NOEXC {
  return -static_cast<s::cl_long>(std::isfinite(x));
}
__SYCL_EXPORT s::cl_int IsFinite(s::cl_half x) __NOEXC {
  return std::isfinite(d::cast_if_host_half(x));
}
__SYCL_EXPORT s::cl_short __vIsFinite(s::cl_half x) __NOEXC {
  return -static_cast<s::cl_int>(std::isfinite(d::cast_if_host_half(x)));
}
MAKE_1V_FUNC(IsFinite, __vIsFinite, s::cl_int, s::cl_float)
MAKE_1V_FUNC(IsFinite, __vIsFinite, s::cl_long, s::cl_double)
MAKE_1V_FUNC(IsFinite, __vIsFinite, s::cl_short, s::cl_half)

// (IsInf)                // isinf
__SYCL_EXPORT s::cl_int IsInf(s::cl_float x) __NOEXC { return std::isinf(x); }
__SYCL_EXPORT s::cl_int IsInf(s::cl_double x) __NOEXC { return std::isinf(x); }
__SYCL_EXPORT s::cl_int __vIsInf(s::cl_float x) __NOEXC {
  return -static_cast<s::cl_int>(std::isinf(x));
}
__SYCL_EXPORT s::cl_long __vIsInf(s::cl_double x) __NOEXC {
  return -static_cast<s::cl_long>(std::isinf(x));
}
__SYCL_EXPORT s::cl_int IsInf(s::cl_half x) __NOEXC {
  return std::isinf(d::cast_if_host_half(x));
}
__SYCL_EXPORT s::cl_short __vIsInf(s::cl_half x) __NOEXC {
  return -static_cast<s::cl_short>(std::isinf(d::cast_if_host_half(x)));
}
MAKE_1V_FUNC(IsInf, __vIsInf, s::cl_int, s::cl_float)
MAKE_1V_FUNC(IsInf, __vIsInf, s::cl_long, s::cl_double)
MAKE_1V_FUNC(IsInf, __vIsInf, s::cl_short, s::cl_half)

// (IsNan)                // isnan
__SYCL_EXPORT s::cl_int IsNan(s::cl_float x) __NOEXC { return std::isnan(x); }
__SYCL_EXPORT s::cl_int IsNan(s::cl_double x) __NOEXC { return std::isnan(x); }
__SYCL_EXPORT s::cl_int __vIsNan(s::cl_float x) __NOEXC {
  return -static_cast<s::cl_int>(std::isnan(x));
}
__SYCL_EXPORT s::cl_long __vIsNan(s::cl_double x) __NOEXC {
  return -static_cast<s::cl_long>(std::isnan(x));
}

__SYCL_EXPORT s::cl_int IsNan(s::cl_half x) __NOEXC {
  return std::isnan(d::cast_if_host_half(x));
}
__SYCL_EXPORT s::cl_short __vIsNan(s::cl_half x) __NOEXC {
  return -static_cast<s::cl_short>(std::isnan(d::cast_if_host_half(x)));
}
MAKE_1V_FUNC(IsNan, __vIsNan, s::cl_int, s::cl_float)
MAKE_1V_FUNC(IsNan, __vIsNan, s::cl_long, s::cl_double)
MAKE_1V_FUNC(IsNan, __vIsNan, s::cl_short, s::cl_half)

// (IsNormal)             // isnormal
__SYCL_EXPORT s::cl_int IsNormal(s::cl_float x) __NOEXC {
  return std::isnormal(x);
}
__SYCL_EXPORT s::cl_int IsNormal(s::cl_double x) __NOEXC {
  return std::isnormal(x);
}
__SYCL_EXPORT s::cl_int __vIsNormal(s::cl_float x) __NOEXC {
  return -static_cast<s::cl_int>(std::isnormal(x));
}
__SYCL_EXPORT s::cl_long __vIsNormal(s::cl_double x) __NOEXC {
  return -static_cast<s::cl_long>(std::isnormal(x));
}
__SYCL_EXPORT s::cl_int IsNormal(s::cl_half x) __NOEXC {
  return std::isnormal(d::cast_if_host_half(x));
}
__SYCL_EXPORT s::cl_short __vIsNormal(s::cl_half x) __NOEXC {
  return -static_cast<s::cl_short>(std::isnormal(d::cast_if_host_half(x)));
}
MAKE_1V_FUNC(IsNormal, __vIsNormal, s::cl_int, s::cl_float)
MAKE_1V_FUNC(IsNormal, __vIsNormal, s::cl_long, s::cl_double)
MAKE_1V_FUNC(IsNormal, __vIsNormal, s::cl_short, s::cl_half)

// (Ordered)              // isordered
__SYCL_EXPORT s::cl_int Ordered(s::cl_float x, s::cl_float y) __NOEXC {
  return __sOrdered(x, y);
}
__SYCL_EXPORT s::cl_int Ordered(s::cl_double x, s::cl_double y) __NOEXC {
  return __sOrdered(x, y);
}
__SYCL_EXPORT s::cl_int Ordered(s::cl_half x, s::cl_half y) __NOEXC {
  return __sOrdered(x, y);
}
MAKE_1V_2V_FUNC(Ordered, __vOrdered, s::cl_int, s::cl_float, s::cl_float)
MAKE_1V_2V_FUNC(Ordered, __vOrdered, s::cl_long, s::cl_double, s::cl_double)
MAKE_1V_2V_FUNC(Ordered, __vOrdered, s::cl_short, s::cl_half, s::cl_half)

// (Unordered)            // isunordered
__SYCL_EXPORT s::cl_int Unordered(s::cl_float x, s::cl_float y) __NOEXC {
  return __sUnordered(x, y);
}
__SYCL_EXPORT s::cl_int Unordered(s::cl_double x, s::cl_double y) __NOEXC {
  return __sUnordered(x, y);
}
__SYCL_EXPORT s::cl_int Unordered(s::cl_half x, s::cl_half y) __NOEXC {
  return __sUnordered(x, y);
}
MAKE_1V_2V_FUNC(Unordered, __vUnordered, s::cl_int, s::cl_float, s::cl_float)
MAKE_1V_2V_FUNC(Unordered, __vUnordered, s::cl_long, s::cl_double, s::cl_double)
MAKE_1V_2V_FUNC(Unordered, __vUnordered, s::cl_short, s::cl_half, s::cl_half)

// (SignBitSet)           // signbit
__SYCL_EXPORT s::cl_int SignBitSet(s::cl_float x) __NOEXC {
  return std::signbit(x);
}
__SYCL_EXPORT s::cl_int SignBitSet(s::cl_double x) __NOEXC {
  return std::signbit(x);
}
__SYCL_EXPORT s::cl_int __vSignBitSet(s::cl_float x) __NOEXC {
  return -static_cast<s::cl_int>(std::signbit(x));
}
__SYCL_EXPORT s::cl_long __vSignBitSet(s::cl_double x) __NOEXC {
  return -static_cast<s::cl_long>(std::signbit(x));
}
__SYCL_EXPORT s::cl_int SignBitSet(s::cl_half x) __NOEXC {
  return std::signbit(d::cast_if_host_half(x));
}
__SYCL_EXPORT s::cl_short __vSignBitSet(s::cl_half x) __NOEXC {
  return -static_cast<s::cl_short>(std::signbit(d::cast_if_host_half(x)));
}
MAKE_1V_FUNC(SignBitSet, __vSignBitSet, s::cl_int, s::cl_float)
MAKE_1V_FUNC(SignBitSet, __vSignBitSet, s::cl_long, s::cl_double)
MAKE_1V_FUNC(SignBitSet, __vSignBitSet, s::cl_short, s::cl_half)

// (Any)                  // any
MAKE_SR_1V_OR(Any, __Any, s::cl_int, s::cl_char)
MAKE_SR_1V_OR(Any, __Any, s::cl_int, s::cl_short)
MAKE_SR_1V_OR(Any, __Any, s::cl_int, s::cl_int)
MAKE_SR_1V_OR(Any, __Any, s::cl_int, s::cl_long)

// (All)                  // all
MAKE_SR_1V_AND(All, __All, s::cl_int, s::cl_char)
MAKE_SR_1V_AND(All, __All, s::cl_int, s::cl_short)
MAKE_SR_1V_AND(All, __All, s::cl_int, s::cl_int)
MAKE_SR_1V_AND(All, __All, s::cl_int, s::cl_long)

// (bitselect)
// Instantiate functions for the scalar types and vector types.
MAKE_SC_1V_2V_3V(bitselect, s::cl_float, s::cl_float, s::cl_float, s::cl_float)
MAKE_SC_1V_2V_3V(bitselect, s::cl_double, s::cl_double, s::cl_double,
                 s::cl_double)
MAKE_SC_1V_2V_3V(bitselect, s::cl_char, s::cl_char, s::cl_char, s::cl_char)
MAKE_SC_1V_2V_3V(bitselect, s::cl_uchar, s::cl_uchar, s::cl_uchar, s::cl_uchar)
MAKE_SC_1V_2V_3V(bitselect, s::cl_short, s::cl_short, s::cl_short, s::cl_short)
MAKE_SC_1V_2V_3V(bitselect, s::cl_ushort, s::cl_ushort, s::cl_ushort,
                 s::cl_ushort)
MAKE_SC_1V_2V_3V(bitselect, s::cl_int, s::cl_int, s::cl_int, s::cl_int)
MAKE_SC_1V_2V_3V(bitselect, s::cl_uint, s::cl_uint, s::cl_uint, s::cl_uint)
MAKE_SC_1V_2V_3V(bitselect, s::cl_long, s::cl_long, s::cl_long, s::cl_long)
MAKE_SC_1V_2V_3V(bitselect, s::cl_ulong, s::cl_ulong, s::cl_ulong, s::cl_ulong)
MAKE_SC_1V_2V_3V(bitselect, s::cl_half, s::cl_half, s::cl_half, s::cl_half)

// (Select) // select
// for scalar: result = c ? b : a.
// for vector: result[i] = (MSB of c[i] is set)? b[i] : a[i]
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_float, s::cl_float,
                        s::cl_float, s::cl_int)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_float, s::cl_float,
                        s::cl_float, s::cl_uint)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_double, s::cl_double,
                        s::cl_double, s::cl_long)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_double, s::cl_double,
                        s::cl_double, s::cl_ulong)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_char, s::cl_char, s::cl_char,
                        s::cl_char)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_char, s::cl_char, s::cl_char,
                        s::cl_uchar)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_uchar, s::cl_uchar,
                        s::cl_uchar, s::cl_char)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_uchar, s::cl_uchar,
                        s::cl_uchar, s::cl_uchar)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_short, s::cl_short,
                        s::cl_short, s::cl_short)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_short, s::cl_short,
                        s::cl_short, s::cl_ushort)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_ushort, s::cl_ushort,
                        s::cl_ushort, s::cl_short)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_ushort, s::cl_ushort,
                        s::cl_ushort, s::cl_ushort)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_int, s::cl_int, s::cl_int,
                        s::cl_int)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_int, s::cl_int, s::cl_int,
                        s::cl_uint)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_uint, s::cl_uint, s::cl_uint,
                        s::cl_int)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_uint, s::cl_uint, s::cl_uint,
                        s::cl_uint)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_long, s::cl_long, s::cl_long,
                        s::cl_long)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_long, s::cl_long, s::cl_long,
                        s::cl_ulong)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_ulong, s::cl_ulong,
                        s::cl_ulong, s::cl_long)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_ulong, s::cl_ulong,
                        s::cl_ulong, s::cl_ulong)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_half, s::cl_half, s::cl_half,
                        s::cl_short)
MAKE_SC_FSC_1V_2V_3V_FV(select, __vselect, s::cl_half, s::cl_half, s::cl_half,
                        s::cl_ushort)
} // namespace __host_std
} // __SYCL_INLINE_NAMESPACE(cl)
