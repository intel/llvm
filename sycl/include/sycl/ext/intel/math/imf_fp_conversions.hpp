//==--------- imf_fp_conversions.hpp - floating point conversions ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// APIs for floating point conversions
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/bit_cast.hpp>
#include <sycl/ext/intel/math.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/half_type.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::math {

/// --------------------------------------------------------------------------
/// Float to integral conversions
/// Supports rounding down/up/to-nearest-even/towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL int __imf_float2int_rd(float);
__DPCPP_SYCL_EXTERNAL int __imf_float2int_rn(float);
__DPCPP_SYCL_EXTERNAL int __imf_float2int_ru(float);
__DPCPP_SYCL_EXTERNAL int __imf_float2int_rz(float);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rd(float);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rn(float);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_ru(float);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_float2uint_rz(float);
__DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rd(float);
__DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rn(float);
__DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_ru(float);
__DPCPP_SYCL_EXTERNAL long long int __imf_float2ll_rz(float);
__DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rd(float);
__DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rn(float);
__DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_ru(float);
__DPCPP_SYCL_EXTERNAL unsigned long long int __imf_float2ull_rz(float);
}

template <typename To = int, typename From = float> To float2int_rd(From x) {
  return __imf_float2int_rd(x);
}

template <typename To = int, typename From = float> To float2int_rn(From x) {
  return __imf_float2int_rn(x);
}

template <typename To = int, typename From = float> To float2int_ru(From x) {
  return __imf_float2int_ru(x);
}

template <typename To = int, typename From = float> To float2int_rz(From x) {
  return __imf_float2int_rz(x);
}

template <typename To = unsigned int, typename From = float>
To float2uint_rd(From x) {
  return __imf_float2uint_rd(x);
}

template <typename To = unsigned int, typename From = float>
To float2uint_rn(From x) {
  return __imf_float2uint_rn(x);
}

template <typename To = unsigned int, typename From = float>
To float2uint_ru(From x) {
  return __imf_float2uint_ru(x);
}

template <typename To = unsigned int, typename From = float>
To float2uint_rz(From x) {
  return __imf_float2uint_rz(x);
}

template <typename To = long long, typename From = float>
To float2ll_rd(From x) {
  return __imf_float2ll_rd(x);
}

template <typename To = long long, typename From = float>
To float2ll_rn(From x) {
  return __imf_float2ll_rn(x);
}

template <typename To = long long, typename From = float>
To float2ll_ru(From x) {
  return __imf_float2ll_ru(x);
}

template <typename To = long long, typename From = float>
To float2ll_rz(From x) {
  return __imf_float2ll_rz(x);
}

template <typename To = unsigned long long, typename From = float>
To float2ull_rd(From x) {
  return __imf_float2ull_rd(x);
}

template <typename To = unsigned long long, typename From = float>
To float2ull_rn(From x) {
  return __imf_float2ull_rn(x);
}

template <typename To = unsigned long long, typename From = float>
To float2ull_ru(From x) {
  return __imf_float2ull_ru(x);
}

template <typename To = unsigned long long, typename From = float>
To float2ull_rz(From x) {
  return __imf_float2ull_rz(x);
}

/// --------------------------------------------------------------------------
/// Integral to float conversions
/// Supports rounding down/up/to-nearest-even/towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL float __imf_int2float_rd(int);
__DPCPP_SYCL_EXTERNAL float __imf_int2float_rn(int);
__DPCPP_SYCL_EXTERNAL float __imf_int2float_ru(int);
__DPCPP_SYCL_EXTERNAL float __imf_int2float_rz(int);
__DPCPP_SYCL_EXTERNAL float __imf_ll2float_rd(long long int);
__DPCPP_SYCL_EXTERNAL float __imf_ll2float_rn(long long int);
__DPCPP_SYCL_EXTERNAL float __imf_ll2float_ru(long long int);
__DPCPP_SYCL_EXTERNAL float __imf_ll2float_rz(long long int);
__DPCPP_SYCL_EXTERNAL float __imf_uint2float_rd(unsigned int);
__DPCPP_SYCL_EXTERNAL float __imf_uint2float_rn(unsigned int);
__DPCPP_SYCL_EXTERNAL float __imf_uint2float_ru(unsigned int);
__DPCPP_SYCL_EXTERNAL float __imf_uint2float_rz(unsigned int);
__DPCPP_SYCL_EXTERNAL float __imf_ull2float_rd(unsigned long long int);
__DPCPP_SYCL_EXTERNAL float __imf_ull2float_rn(unsigned long long int);
__DPCPP_SYCL_EXTERNAL float __imf_ull2float_ru(unsigned long long int);
__DPCPP_SYCL_EXTERNAL float __imf_ull2float_rz(unsigned long long int);
}

template <typename To = float, typename From = long long>
To ll2float_rd(From x) {
  return __imf_ll2float_rd(x);
}

template <typename To = float, typename From = long long>
To ll2float_rn(From x) {
  return __imf_ll2float_rn(x);
}

template <typename To = float, typename From = long long>
To ll2float_ru(From x) {
  return __imf_ll2float_ru(x);
}

template <typename To = float, typename From = long long>
To ll2float_rz(From x) {
  return __imf_ll2float_rz(x);
}

template <typename To = float, typename From = unsigned long long>
To ull2float_rd(From x) {
  return __imf_ull2float_rd(x);
}

template <typename To = float, typename From = unsigned long long>
To ull2float_rn(From x) {
  return __imf_ull2float_rn(x);
}

template <typename To = float, typename From = unsigned long long>
To ull2float_ru(From x) {
  return __imf_ull2float_ru(x);
}

template <typename To = float, typename From = unsigned long long>
To ull2float_rz(From x) {
  return __imf_ull2float_rz(x);
}

template <typename To = float, typename From = int> To int2float_rd(From x) {
  return __imf_int2float_rd(x);
}

template <typename To = float, typename From = int> To int2float_rn(From x) {
  return __imf_int2float_rn(x);
}

template <typename To = float, typename From = int> To int2float_ru(From x) {
  return __imf_int2float_ru(x);
}

template <typename To = float, typename From = int> To int2float_rz(From x) {
  return __imf_int2float_rz(x);
}

template <typename To = float, typename From = unsigned int>
To uint2float_rd(From x) {
  return __imf_uint2float_rd(x);
}

template <typename To = float, typename From = unsigned int>
To uint2float_rn(From x) {
  return __imf_uint2float_rn(x);
}

template <typename To = float, typename From = unsigned int>
To uint2float_ru(From x) {
  return __imf_uint2float_ru(x);
}

template <typename To = float, typename From = unsigned int>
To uint2float_rz(From x) {
  return __imf_uint2float_rz(x);
}

/// --------------------------------------------------------------------------
/// Reinterprets bits in float type as int/unsigned int type and vice versa.
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL int __imf_float_as_int(float);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_float_as_uint(float);
__DPCPP_SYCL_EXTERNAL float __imf_int_as_float(int);
__DPCPP_SYCL_EXTERNAL float __imf_uint_as_float(unsigned int);
}

template <typename To = int, typename From = float> To float_as_int(From x) {
  return __imf_float_as_int(x);
}

template <typename To = unsigned int, typename From = float>
To float_as_uint(From x) {
  return __imf_float_as_uint(x);
}

template <typename To = float, typename From = int> To int_as_float(From x) {
  return __imf_int_as_float(x);
}

template <typename To = float, typename From = unsigned int>
To uint_as_float(From x) {
  return __imf_uint_as_float(x);
}

/// --------------------------------------------------------------------------
/// Double to float conversions
/// Supports rounding down/up/to-nearest-even/towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL float __imf_double2float_rd(double);
__DPCPP_SYCL_EXTERNAL float __imf_double2float_rn(double);
__DPCPP_SYCL_EXTERNAL float __imf_double2float_ru(double);
__DPCPP_SYCL_EXTERNAL float __imf_double2float_rz(double);
}

template <typename To = float, typename From = double>
To double2float_rd(From x) {
  return __imf_double2float_rd(x);
}

template <typename To = float, typename From = double>
To double2float_rn(From x) {
  return __imf_double2float_rn(x);
}

template <typename To = float, typename From = double>
To double2float_ru(From x) {
  return __imf_double2float_ru(x);
}

template <typename To = float, typename From = double>
To double2float_rz(From x) {
  return __imf_double2float_rz(x);
}

/// --------------------------------------------------------------------------
/// Reinterprets high/low 32 bits in a double as a signed integer.
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL int __imf_double2hiint(double);
__DPCPP_SYCL_EXTERNAL int __imf_double2loint(double);
}

template <typename To = int, typename From = double> To double2hiint(From x) {
  return __imf_double2hiint(x);
}

template <typename To = int, typename From = double> To double2loint(From x) {
  return __imf_double2loint(x);
}

/// --------------------------------------------------------------------------
/// Double to integral conversions
/// Supports rounding down/up/to-nearest-even/towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL int __imf_double2int_rd(double);
__DPCPP_SYCL_EXTERNAL int __imf_double2int_rn(double);
__DPCPP_SYCL_EXTERNAL int __imf_double2int_ru(double);
__DPCPP_SYCL_EXTERNAL int __imf_double2int_rz(double);
__DPCPP_SYCL_EXTERNAL long long __imf_double2ll_rd(double);
__DPCPP_SYCL_EXTERNAL long long __imf_double2ll_rn(double);
__DPCPP_SYCL_EXTERNAL long long __imf_double2ll_ru(double);
__DPCPP_SYCL_EXTERNAL long long __imf_double2ll_rz(double);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rd(double);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rn(double);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_ru(double);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_double2uint_rz(double);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_double2ull_rd(double);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_double2ull_rn(double);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_double2ull_ru(double);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_double2ull_rz(double);
}

template <typename To = int, typename From = double> To double2int_rd(From x) {
  return __imf_double2int_rd(x);
}

template <typename To = int, typename From = double> To double2int_rn(From x) {
  return __imf_double2int_rn(x);
}

template <typename To = int, typename From = double> To double2int_ru(From x) {
  return __imf_double2int_ru(x);
}

template <typename To = int, typename From = double> To double2int_rz(From x) {
  return __imf_double2int_rz(x);
}

template <typename To = long long, typename From = double>
To double2ll_rd(From x) {
  return __imf_double2ll_rd(x);
}

template <typename To = long long, typename From = double>
To double2ll_rn(From x) {
  return __imf_double2ll_rn(x);
}

template <typename To = long long, typename From = double>
To double2ll_ru(From x) {
  return __imf_double2ll_ru(x);
}

template <typename To = long long, typename From = double>
To double2ll_rz(From x) {
  return __imf_double2ll_rz(x);
}

template <typename To = unsigned int, typename From = double>
To double2uint_rd(From x) {
  return __imf_double2uint_rd(x);
}

template <typename To = unsigned int, typename From = double>
To double2uint_rn(From x) {
  return __imf_double2uint_rn(x);
}

template <typename To = unsigned int, typename From = double>
To double2uint_ru(From x) {
  return __imf_double2uint_ru(x);
}

template <typename To = unsigned int, typename From = double>
To double2uint_rz(From x) {
  return __imf_double2uint_rz(x);
}

template <typename To = unsigned long long, typename From = double>
To double2ull_rd(From x) {
  return __imf_double2ull_rd(x);
}

template <typename To = unsigned long long, typename From = double>
To double2ull_rn(From x) {
  return __imf_double2ull_rn(x);
}

template <typename To = unsigned long long, typename From = double>
To double2ull_ru(From x) {
  return __imf_double2ull_ru(x);
}

template <typename To = unsigned long long, typename From = double>
To double2ull_rz(From x) {
  return __imf_double2ull_rz(x);
}

/// --------------------------------------------------------------------------
/// Reinterprets bits in double as long long and vice versa.
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL long long __imf_double_as_longlong(double);
__DPCPP_SYCL_EXTERNAL double __imf_longlong_as_double(long long);
}

template <typename To = long long, typename From = double>
To double_as_longlong(From x) {
  return __imf_double_as_longlong(x);
}

template <typename To = double, typename From = long long>
To longlong_as_double(From x) {
  return __imf_longlong_as_double(x);
}

/// --------------------------------------------------------------------------
/// Reinterprets 2 32-bit integers as high and low 32-bits in a double value.
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL double __imf_hiloint2double(int, int);
}

template <typename To = double, typename From = int>
To hiloint2double(From x, From y) {
  return __imf_hiloint2double(x, y);
}

/// --------------------------------------------------------------------------
/// Signed and unsigned int type to double conversions in rounding towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL double __imf_int2double_rn(int);
__DPCPP_SYCL_EXTERNAL double __imf_uint2double_rn(unsigned);
}

template <typename To = double, typename From = int> To int2double_rn(From x) {
  return __imf_int2double_rn(x);
}

template <typename To = double, typename From = unsigned int>
To uint2double_rn(From x) {
  return __imf_uint2double_rn(x);
}

/// --------------------------------------------------------------------------
/// Signed and unsigned long long type to double conversions
/// Supports rounding down/up/to-nearest-even/towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL double __imf_ll2double_rd(long long);
__DPCPP_SYCL_EXTERNAL double __imf_ll2double_rn(long long);
__DPCPP_SYCL_EXTERNAL double __imf_ll2double_ru(long long);
__DPCPP_SYCL_EXTERNAL double __imf_ll2double_rz(long long);
__DPCPP_SYCL_EXTERNAL double __imf_ull2double_rd(unsigned long long);
__DPCPP_SYCL_EXTERNAL double __imf_ull2double_rn(unsigned long long);
__DPCPP_SYCL_EXTERNAL double __imf_ull2double_ru(unsigned long long);
__DPCPP_SYCL_EXTERNAL double __imf_ull2double_rz(unsigned long long);
}

template <typename To = double, typename From = long long>
To ll2double_rd(From x) {
  return __imf_ll2double_rd(x);
}

template <typename To = double, typename From = long long>
To ll2double_rn(From x) {
  return __imf_ll2double_rn(x);
}

template <typename To = double, typename From = long long>
To ll2double_ru(From x) {
  return __imf_ll2double_ru(x);
}

template <typename To = double, typename From = long long>
To ll2double_rz(From x) {
  return __imf_ll2double_rz(x);
}

template <typename To = double, typename From = unsigned long long>
To ull2double_rd(From x) {
  return __imf_ull2double_rd(x);
}

template <typename To = double, typename From = unsigned long long>
To ull2double_rn(From x) {
  return __imf_ull2double_rn(x);
}

template <typename To = double, typename From = unsigned long long>
To ull2double_ru(From x) {
  return __imf_ull2double_ru(x);
}

template <typename To = double, typename From = unsigned long long>
To ull2double_rz(From x) {
  return __imf_ull2double_rz(x);
}

/// --------------------------------------------------------------------------
/// Half to float conversions
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL float __imf_half2float(_iml_half_internal);
}

template <typename To = float, typename From = sycl::half>
To half2float(From x) {
  return __imf_half2float(sycl::bit_cast<_iml_half_internal>(x));
}

/// --------------------------------------------------------------------------
/// Float to half conversions
/// Supports rounding down/up/to-nearest-even/towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_float2half_rd(float);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_float2half_rn(float);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_float2half_ru(float);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_float2half_rz(float);
}

template <typename To = sycl::half, typename From = float>
To float2half_rn(From x) {
  return sycl::bit_cast<sycl::half>(__imf_float2half_rn(x));
}

template <typename To = sycl::half, typename From = float>
To float2half_rd(From x) {
  return sycl::bit_cast<sycl::half>(__imf_float2half_rd(x));
}

template <typename To = sycl::half, typename From = float>
To float2half_ru(From x) {
  return sycl::bit_cast<sycl::half>(__imf_float2half_ru(x));
}

template <typename To = sycl::half, typename From = float>
To float2half_rz(From x) {
  return sycl::bit_cast<sycl::half>(__imf_float2half_rz(x));
}

/// --------------------------------------------------------------------------
/// Double to half conversions in rounding to-nearest-even.
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_double2half(double);
}

template <typename To = sycl::half, typename From = double>
To double2half(From x) {
  return sycl::bit_cast<sycl::half>(__imf_double2half(x));
}

/// --------------------------------------------------------------------------
/// Half to integral type conversions
/// Supports rounding down/up/to-nearest-even/towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL int __imf_half2int_rd(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL int __imf_half2int_rn(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL int __imf_half2int_ru(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL int __imf_half2int_rz(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rd(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rn(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL long long __imf_half2ll_ru(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL long long __imf_half2ll_rz(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL short __imf_half2short_rd(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL short __imf_half2short_rn(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL short __imf_half2short_ru(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL short __imf_half2short_rz(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rd(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rn(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_ru(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_half2uint_rz(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rd(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rn(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_ru(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_half2ull_rz(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rd(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rn(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_ru(_iml_half_internal);
__DPCPP_SYCL_EXTERNAL unsigned short __imf_half2ushort_rz(_iml_half_internal);
}

template <typename To = int, typename From = sycl::half>
To half2int_rn(From x) {
  return __imf_half2int_rn(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = int, typename From = sycl::half>
To half2int_rd(From x) {
  return __imf_half2int_rd(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = int, typename From = sycl::half>
To half2int_ru(From x) {
  return __imf_half2int_ru(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = int, typename From = sycl::half>
To half2int_rz(From x) {
  return __imf_half2int_rz(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = long long, typename From = sycl::half>
To half2ll_rn(From x) {
  return __imf_half2ll_rn(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = long long, typename From = sycl::half>
To half2ll_rd(From x) {
  return __imf_half2ll_rd(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = long long, typename From = sycl::half>
To half2ll_ru(From x) {
  return __imf_half2ll_ru(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = long long, typename From = sycl::half>
To half2ll_rz(From x) {
  return __imf_half2ll_rz(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = short, typename From = sycl::half>
To half2short_rn(From x) {
  return __imf_half2short_rn(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = short, typename From = sycl::half>
To half2short_rd(From x) {
  return __imf_half2short_rd(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = short, typename From = sycl::half>
To half2short_ru(From x) {
  return __imf_half2short_ru(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = short, typename From = sycl::half>
To half2short_rz(From x) {
  return __imf_half2short_rz(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned short, typename From = sycl::half>
To half2ushort_rn(From x) {
  return __imf_half2ushort_rn(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned short, typename From = sycl::half>
To half2ushort_rd(From x) {
  return __imf_half2ushort_rd(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned short, typename From = sycl::half>
To half2ushort_ru(From x) {
  return __imf_half2ushort_ru(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned short, typename From = sycl::half>
To half2ushort_rz(From x) {
  return __imf_half2ushort_rz(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned int, typename From = sycl::half>
To half2uint_rn(From x) {
  return __imf_half2uint_rn(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned int, typename From = sycl::half>
To half2uint_rd(From x) {
  return __imf_half2uint_rd(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned int, typename From = sycl::half>
To half2uint_ru(From x) {
  return __imf_half2uint_ru(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned int, typename From = sycl::half>
To half2uint_rz(From x) {
  return __imf_half2uint_rz(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned long long, typename From = sycl::half>
To half2ull_rn(From x) {
  return __imf_half2ull_rn(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned long long, typename From = sycl::half>
To half2ull_rd(From x) {
  return __imf_half2ull_rd(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned long long, typename From = sycl::half>
To half2ull_ru(From x) {
  return __imf_half2ull_ru(sycl::bit_cast<_iml_half_internal>(x));
}

template <typename To = unsigned long long, typename From = sycl::half>
To half2ull_rz(From x) {
  return __imf_half2ull_rz(sycl::bit_cast<_iml_half_internal>(x));
}

/// --------------------------------------------------------------------------
/// Integral type to half conversions
/// Supports rounding down/up/to-nearest-even/towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_int2half_rd(int);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_int2half_rn(int);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_int2half_ru(int);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_int2half_rz(int);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ll2half_rd(long long);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ll2half_rn(long long);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ll2half_ru(long long);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ll2half_rz(long long);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_short2half_rd(short);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_short2half_rn(short);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_short2half_ru(short);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_short2half_rz(short);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_uint2half_rd(unsigned int);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_uint2half_rn(unsigned int);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_uint2half_ru(unsigned int);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_uint2half_rz(unsigned int);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ull2half_rd(unsigned long long);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ull2half_rn(unsigned long long);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ull2half_ru(unsigned long long);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ull2half_rz(unsigned long long);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ushort2half_rd(unsigned short);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ushort2half_rn(unsigned short);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ushort2half_ru(unsigned short);
__DPCPP_SYCL_EXTERNAL _iml_half_internal __imf_ushort2half_rz(unsigned short);
}

template <typename To = sycl::half, typename From = int>
To int2half_rn(From x) {
  return sycl::bit_cast<sycl::half>(__imf_int2half_rn(x));
}

template <typename To = sycl::half, typename From = int>
To int2half_rd(From x) {
  return sycl::bit_cast<sycl::half>(__imf_int2half_rd(x));
}

template <typename To = sycl::half, typename From = int>
To int2half_ru(From x) {
  return sycl::bit_cast<sycl::half>(__imf_int2half_ru(x));
}

template <typename To = sycl::half, typename From = int>
To int2half_rz(From x) {
  return sycl::bit_cast<sycl::half>(__imf_int2half_rz(x));
}

template <typename To = sycl::half, typename From = short>
To short2half_rn(From x) {
  return sycl::bit_cast<sycl::half>(__imf_short2half_rn(x));
}

template <typename To = sycl::half, typename From = short>
To short2half_rd(From x) {
  return sycl::bit_cast<sycl::half>(__imf_short2half_rd(x));
}

template <typename To = sycl::half, typename From = short>
To short2half_ru(From x) {
  return sycl::bit_cast<sycl::half>(__imf_short2half_ru(x));
}

template <typename To = sycl::half, typename From = short>
To short2half_rz(From x) {
  return sycl::bit_cast<sycl::half>(__imf_short2half_rz(x));
}

template <typename To = sycl::half, typename From = long long>
To ll2half_rn(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ll2half_rn(x));
}

template <typename To = sycl::half, typename From = long long>
To ll2half_rd(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ll2half_rd(x));
}

template <typename To = sycl::half, typename From = long long>
To ll2half_ru(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ll2half_ru(x));
}

template <typename To = sycl::half, typename From = long long>
To ll2half_rz(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ll2half_rz(x));
}

template <typename To = sycl::half, typename From = unsigned short>
To ushort2half_rn(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ushort2half_rn(x));
}

template <typename To = sycl::half, typename From = unsigned short>
To ushort2half_rd(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ushort2half_rd(x));
}

template <typename To = sycl::half, typename From = unsigned short>
To ushort2half_ru(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ushort2half_ru(x));
}

template <typename To = sycl::half, typename From = unsigned short>
To ushort2half_rz(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ushort2half_rz(x));
}

template <typename To = sycl::half, typename From = unsigned int>
To uint2half_rn(From x) {
  return sycl::bit_cast<sycl::half>(__imf_uint2half_rn(x));
}

template <typename To = sycl::half, typename From = unsigned int>
To uint2half_rd(From x) {
  return sycl::bit_cast<sycl::half>(__imf_uint2half_rd(x));
}

template <typename To = sycl::half, typename From = unsigned int>
To uint2half_ru(From x) {
  return sycl::bit_cast<sycl::half>(__imf_uint2half_ru(x));
}

template <typename To = sycl::half, typename From = unsigned int>
To uint2half_rz(From x) {
  return sycl::bit_cast<sycl::half>(__imf_uint2half_rz(x));
}

template <typename To = sycl::half, typename From = unsigned long long>
To ull2half_rn(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ull2half_rn(x));
}

template <typename To = sycl::half, typename From = unsigned long long>
To ull2half_rd(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ull2half_rd(x));
}

template <typename To = sycl::half, typename From = unsigned long long>
To ull2half_ru(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ull2half_ru(x));
}

template <typename To = sycl::half, typename From = unsigned long long>
To ull2half_rz(From x) {
  return sycl::bit_cast<sycl::half>(__imf_ull2half_rz(x));
}

/// --------------------------------------------------------------------------
/// Reinterprets bits in bfloat16 type as short/unsigned short and vice versa.
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL short __imf_bfloat16_as_short(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned short __imf_bfloat16_as_ushort(uint16_t);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_short_as_bfloat16(short);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ushort_as_bfloat16(unsigned short);
}

template <typename To = unsigned short,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat16_as_ushort(From x) {
  return __imf_bfloat16_as_ushort(sycl::bit_cast<uint16_t>(x));
}

template <typename To = short, typename From = sycl::ext::oneapi::bfloat16>
To bfloat16_as_short(From x) {
  return __imf_bfloat16_as_short(sycl::bit_cast<uint16_t>(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned short>
To ushort_as_bfloat16(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_ushort_as_bfloat16(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = short>
To short_as_bfloat16(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_short_as_bfloat16(x));
}

/// --------------------------------------------------------------------------
/// Bfloat16 to float conversions
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL float __imf_bfloat162float(uint16_t);
}

template <typename To = float, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162float(From x) {
  return __imf_bfloat162float(sycl::bit_cast<uint16_t>(x));
}

/// --------------------------------------------------------------------------
/// Float to bfloat16 conversions
/// Supports rounding down/up/to-nearest-even/towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rd(float);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rn(float);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_ru(float);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_float2bfloat16_rz(float);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = float>
To float2bfloat16(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_float2bfloat16_rn(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = float>
To float2bfloat16_rd(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_float2bfloat16_rd(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = float>
To float2bfloat16_rn(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_float2bfloat16_rn(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = float>
To float2bfloat16_ru(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_float2bfloat16_ru(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = float>
To float2bfloat16_rz(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_float2bfloat16_rz(x));
}

/// --------------------------------------------------------------------------
/// Integral to bfloat16 conversions
/// Supports rounding down/up/to-nearest-even/towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ushort2bfloat16_rd(unsigned short);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ushort2bfloat16_rn(unsigned short);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ushort2bfloat16_ru(unsigned short);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ushort2bfloat16_rz(unsigned short);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rd(unsigned int);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rn(unsigned int);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_ru(unsigned int);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_uint2bfloat16_rz(unsigned int);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ull2bfloat16_rd(unsigned long long);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ull2bfloat16_rn(unsigned long long);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ull2bfloat16_ru(unsigned long long);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ull2bfloat16_rz(unsigned long long);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rd(short);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rn(short);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_ru(short);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_short2bfloat16_rz(short);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rd(int);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rn(int);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_ru(int);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_int2bfloat16_rz(int);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rd(long long);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rn(long long);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_ru(long long);
__DPCPP_SYCL_EXTERNAL uint16_t __imf_ll2bfloat16_rz(long long);
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned short>
To ushort2bfloat16_rd(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_ushort2bfloat16_rd(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned short>
To ushort2bfloat16_rn(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_ushort2bfloat16_rn(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned short>
To ushort2bfloat16_ru(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_ushort2bfloat16_ru(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned short>
To ushort2bfloat16_rz(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_ushort2bfloat16_rz(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned int>
To uint2bfloat16_rd(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_uint2bfloat16_rd(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned int>
To uint2bfloat16_rn(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_uint2bfloat16_rn(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned int>
To uint2bfloat16_ru(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_uint2bfloat16_ru(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned int>
To uint2bfloat16_rz(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_uint2bfloat16_rz(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned long long>
To ull2bfloat16_rd(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_ull2bfloat16_rd(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned long long>
To ull2bfloat16_rn(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_ull2bfloat16_rn(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned long long>
To ull2bfloat16_ru(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_ull2bfloat16_ru(x));
}

template <typename To = sycl::ext::oneapi::bfloat16,
          typename From = unsigned long long>
To ull2bfloat16_rz(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_ull2bfloat16_rz(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = short>
To short2bfloat16_rd(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_short2bfloat16_rd(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = short>
To short2bfloat16_rn(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_short2bfloat16_rn(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = short>
To short2bfloat16_ru(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_short2bfloat16_ru(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = short>
To short2bfloat16_rz(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(
      __imf_short2bfloat16_rz(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = int>
To int2bfloat16_rd(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_int2bfloat16_rd(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = int>
To int2bfloat16_rn(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_int2bfloat16_rn(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = int>
To int2bfloat16_ru(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_int2bfloat16_ru(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = int>
To int2bfloat16_rz(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_int2bfloat16_rz(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = long long>
To ll2bfloat16_rd(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_ll2bfloat16_rd(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = long long>
To ll2bfloat16_rn(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_ll2bfloat16_rn(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = long long>
To ll2bfloat16_ru(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_ll2bfloat16_ru(x));
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = long long>
To ll2bfloat16_rz(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_ll2bfloat16_rz(x));
}

/// --------------------------------------------------------------------------
/// Double to bfloat16 conversions in rounding to-nearest-even.
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL uint16_t __imf_double2bfloat16(double);
}

template <typename To = sycl::ext::oneapi::bfloat16, typename From = double>
To double2bfloat16(From x) {
  return sycl::bit_cast<sycl::ext::oneapi::bfloat16>(__imf_double2bfloat16(x));
}

/// --------------------------------------------------------------------------
/// Bfloat16 to integral conversions
/// Supports rounding down/up/to-nearest-even/towards-zero
/// --------------------------------------------------------------------------
extern "C" {
__DPCPP_SYCL_EXTERNAL unsigned short __imf_bfloat162ushort_rd(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned short __imf_bfloat162ushort_rn(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned short __imf_bfloat162ushort_ru(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned short __imf_bfloat162ushort_rz(uint16_t);
__DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rd(uint16_t);
__DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rn(uint16_t);
__DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_ru(uint16_t);
__DPCPP_SYCL_EXTERNAL short __imf_bfloat162short_rz(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rd(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rn(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_ru(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned int __imf_bfloat162uint_rz(uint16_t);
__DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rd(uint16_t);
__DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rn(uint16_t);
__DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_ru(uint16_t);
__DPCPP_SYCL_EXTERNAL int __imf_bfloat162int_rz(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_bfloat162ull_rd(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_bfloat162ull_rn(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_bfloat162ull_ru(uint16_t);
__DPCPP_SYCL_EXTERNAL unsigned long long __imf_bfloat162ull_rz(uint16_t);
__DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rd(uint16_t);
__DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rn(uint16_t);
__DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_ru(uint16_t);
__DPCPP_SYCL_EXTERNAL long long __imf_bfloat162ll_rz(uint16_t);
}

template <typename To = unsigned int,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162uint_rd(From x) {
  return __imf_bfloat162uint_rd(sycl::bit_cast<uint16_t>(x));
}

template <typename To = unsigned int,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162uint_rn(From x) {
  return __imf_bfloat162uint_rn(sycl::bit_cast<uint16_t>(x));
}

template <typename To = unsigned int,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162uint_ru(From x) {
  return __imf_bfloat162uint_ru(sycl::bit_cast<uint16_t>(x));
}

template <typename To = unsigned int,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162uint_rz(From x) {
  return __imf_bfloat162uint_rz(sycl::bit_cast<uint16_t>(x));
}

template <typename To = int, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162int_rd(From x) {
  return __imf_bfloat162int_rd(sycl::bit_cast<uint16_t>(x));
}

template <typename To = int, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162int_rn(From x) {
  return __imf_bfloat162int_rn(sycl::bit_cast<uint16_t>(x));
}

template <typename To = int, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162int_ru(From x) {
  return __imf_bfloat162int_ru(sycl::bit_cast<uint16_t>(x));
}

template <typename To = int, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162int_rz(From x) {
  return __imf_bfloat162int_rz(sycl::bit_cast<uint16_t>(x));
}

template <typename To = unsigned short,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ushort_rd(From x) {
  return __imf_bfloat162ushort_rd(sycl::bit_cast<uint16_t>(x));
}

template <typename To = unsigned short,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ushort_rn(From x) {
  return __imf_bfloat162ushort_rn(sycl::bit_cast<uint16_t>(x));
}

template <typename To = unsigned short,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ushort_ru(From x) {
  return __imf_bfloat162ushort_ru(sycl::bit_cast<uint16_t>(x));
}

template <typename To = unsigned short,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ushort_rz(From x) {
  return __imf_bfloat162ushort_rz(sycl::bit_cast<uint16_t>(x));
}

template <typename To = short, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162short_rd(From x) {
  return __imf_bfloat162short_rd(sycl::bit_cast<uint16_t>(x));
}

template <typename To = short, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162short_rn(From x) {
  return __imf_bfloat162short_rn(sycl::bit_cast<uint16_t>(x));
}

template <typename To = short, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162short_ru(From x) {
  return __imf_bfloat162short_ru(sycl::bit_cast<uint16_t>(x));
}

template <typename To = short, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162short_rz(From x) {
  return __imf_bfloat162short_rz(sycl::bit_cast<uint16_t>(x));
}

template <typename To = long long, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ll_rd(From x) {
  return __imf_bfloat162ll_rd(sycl::bit_cast<uint16_t>(x));
}

template <typename To = long long, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ll_rn(From x) {
  return __imf_bfloat162ll_rn(sycl::bit_cast<uint16_t>(x));
}

template <typename To = long long, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ll_ru(From x) {
  return __imf_bfloat162ll_ru(sycl::bit_cast<uint16_t>(x));
}

template <typename To = long long, typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ll_rz(From x) {
  return __imf_bfloat162ll_rz(sycl::bit_cast<uint16_t>(x));
}

template <typename To = unsigned long long,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ull_rd(From x) {
  return __imf_bfloat162ull_rd(sycl::bit_cast<uint16_t>(x));
}

template <typename To = unsigned long long,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ull_rn(From x) {
  return __imf_bfloat162ull_rn(sycl::bit_cast<uint16_t>(x));
}

template <typename To = unsigned long long,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ull_ru(From x) {
  return __imf_bfloat162ull_ru(sycl::bit_cast<uint16_t>(x));
}

template <typename To = unsigned long long,
          typename From = sycl::ext::oneapi::bfloat16>
To bfloat162ull_rz(From x) {
  return __imf_bfloat162ull_rz(sycl::bit_cast<uint16_t>(x));
}

} // namespace ext::intel::math
} // namespace _V1
} // namespace sycl
