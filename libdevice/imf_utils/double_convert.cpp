//==-- double_convert.cpp - fallback implementation of double to other type
// convert--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device_imf.hpp"

#ifdef __LIBDEVICE_IMF_ENABLED__

static inline float __double2float_rd(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<float>(x, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_FConvert_Rfloat_rtn(x);
#endif
}

static inline float __double2float_rn(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<float>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_FConvert_Rfloat_rte(x);
#endif
}

static inline float __double2float_ru(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<float>(x, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_FConvert_Rfloat_rtp(x);
#endif
}

static inline float __double2float_rz(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<float>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_FConvert_Rfloat_rtz(x);
#endif
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_double2float_rd(double x) { return __double2float_rd(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_double2float_rn(double x) { return __double2float_rn(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_double2float_ru(double x) { return __double2float_ru(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_double2float_rz(double x) { return __double2float_rz(x); }

static inline int __double2int_rd(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<int>(x, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_rtn(x);
#endif
}

static inline int __double2int_rn(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<int>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_rte(x);
#endif
}

static inline int __double2int_ru(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<int>(x, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_rtp(x);
#endif
}

static inline int __double2int_rz(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<int>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_rtz(x);
#endif
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2int_rd(double x) { return __double2int_rd(x); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2int_rn(double x) { return __double2int_rn(x); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2int_ru(double x) { return __double2int_ru(x); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2int_rz(double x) { return __double2int_rz(x); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2hiint(double x) {
  uint64_t tmp = __bit_cast<uint64_t>(x);
  tmp = tmp >> 32;
  return static_cast<int>(tmp);
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_double2loint(double x) {
  uint64_t tmp = __bit_cast<uint64_t>(x);
  return static_cast<int>(tmp);
}

// __spirv_ConvertFToU_Ruint_rtn/e/p/z have different behaviors
// on CPU and GPU device when input value is negative.
static inline unsigned int __double2uint_rd(double x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<unsigned int>(x, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_rtn(x);
#endif
}

static inline unsigned int __double2uint_rn(double x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<unsigned int>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_rte(x);
#endif
}

static inline unsigned int __double2uint_ru(double x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<unsigned int>(x, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_rtp(x);
#endif
}

static inline unsigned int __double2uint_rz(double x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<unsigned int>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_rtz(x);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_double2uint_rd(double x) {
  return __double2uint_rd(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_double2uint_rn(double x) {
  return __double2uint_rn(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_double2uint_ru(double x) {
  return __double2uint_ru(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_double2uint_rz(double x) {
  return __double2uint_rz(x);
}

static inline long long int __double2ll_rd(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<long long int>(x, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_rtn(x);
#endif
}

static inline long long int __double2ll_rn(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<long long int>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_rte(x);
#endif
}

static inline long long int __double2ll_ru(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<long long int>(x, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_rtp(x);
#endif
}

static inline long long int __double2ll_rz(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<long long int>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_rtz(x);
#endif
}

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_double2ll_rd(double x) {
  return __double2ll_rd(x);
}

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_double2ll_rn(double x) {
  return __double2ll_rn(x);
}

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_double2ll_ru(double x) {
  return __double2ll_ru(x);
}

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_double2ll_rz(double x) {
  return __double2ll_rz(x);
}

static inline unsigned long long int __double2ull_rd(double x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<unsigned long long int>(x, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_rtn(x);
#endif
}

static inline unsigned long long int __double2ull_rn(double x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<unsigned long long int>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_rte(x);
#endif
}

static inline unsigned long long int __double2ull_ru(double x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<unsigned long long int>(x, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_rtp(x);
#endif
}

static inline unsigned long long int __double2ull_rz(double x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __double2Tp_host<unsigned long long int>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_rtz(x);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_double2ull_rd(double x) {
  return __double2ull_rd(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_double2ull_rn(double x) {
  return __double2ull_rn(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_double2ull_ru(double x) {
  return __double2ull_ru(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_double2ull_rz(double x) {
  return __double2ull_rz(x);
}

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_double_as_longlong(double x) {
  return __bit_cast<long long int>(x);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_hiloint2double(int hi, int lo) {
  uint32_t hiu = __bit_cast<uint32_t>(hi);
  uint32_t lou = __bit_cast<uint32_t>(lo);
  uint64_t res_bits = static_cast<uint64_t>(hiu);
  res_bits = res_bits << 32;
  res_bits = res_bits | static_cast<uint64_t>(lou);
  return __bit_cast<double>(res_bits);
}

static inline double __int2double_rn(int x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int, double>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rdouble(x);
#endif
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_int2double_rn(int x) { return __int2double_rn(x); }

static inline double __ll2double_rd(long long int x) {
  int64_t xi64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int64_t, double>(xi64, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rdouble_rtn(xi64);
#endif
}

static inline double __ll2double_rn(long long int x) {
  int64_t xi64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int64_t, double>(xi64, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rdouble_rte(xi64);
#endif
}

static inline double __ll2double_ru(long long int x) {
  int64_t xi64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int64_t, double>(xi64, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rdouble_rtp(xi64);
#endif
}

static inline double __ll2double_rz(long long int x) {
  int64_t xi64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int64_t, double>(xi64, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rdouble_rtz(xi64);
#endif
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ll2double_rd(long long int x) {
  return __ll2double_rd(x);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ll2double_rn(long long int x) {
  return __ll2double_rn(x);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ll2double_ru(long long int x) {
  return __ll2double_ru(x);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ll2double_rz(long long int x) {
  return __ll2double_rz(x);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_longlong_as_double(long long int x) {
  return __bit_cast<double, long long int>(x);
}

static inline double __uint2double_rn(unsigned int x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<unsigned int, double>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rdouble_rte(x);
#endif
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_uint2double_rn(unsigned int x) {
  return __uint2double_rn(x);
}

static inline double __ull2double_rd(unsigned long long int x) {
  uint64_t xui64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<uint64_t, double>(xui64, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rdouble_rtn(xui64);
#endif
}

static inline double __ull2double_rn(unsigned long long int x) {
  uint64_t xui64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<uint64_t, double>(xui64, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rdouble_rte(xui64);
#endif
}

static inline double __ull2double_ru(unsigned long long int x) {
  uint64_t xui64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<uint64_t, double>(xui64, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rdouble_rtp(xui64);
#endif
}

static inline double __ull2double_rz(unsigned long long int x) {
  uint64_t xui64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<uint64_t, double>(xui64, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rdouble_rtz(xui64);
#endif
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ull2double_rd(unsigned long long int x) {
  return __ull2double_rd(x);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ull2double_rn(unsigned long long int x) {
  return __ull2double_rn(x);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ull2double_ru(unsigned long long int x) {
  return __ull2double_ru(x);
}

DEVICE_EXTERN_C_INLINE
double __devicelib_imf_ull2double_rz(unsigned long long int x) {
  return __ull2double_rz(x);
}

DEVICE_EXTERN_C_INLINE
_iml_half_internal __devicelib_imf_double2half(double x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __iml_fp2half<double>(x, __IML_RTE);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_FConvert_Rhalf_rte(x);
#endif
}

DEVICE_EXTERN_C_INLINE
_iml_bf16_internal __devicelib_imf_double2bfloat16(double x) {
  return __double2bfloat16(x);
}
#endif // __LIBDEVICE_IMF_ENABLED__
