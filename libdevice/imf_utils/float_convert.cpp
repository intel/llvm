//==-- float_convert.cpp - fallback implementation of float to other type
// convert--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../device_imf.hpp"

#ifdef __LIBDEVICE_IMF_ENABLED__

static inline int __float2int_rd(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<int>(x, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_rtn(x);
#endif
}

static inline int __float2int_rn(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<int>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_rte(x);
#endif
}

static inline int __float2int_ru(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<int>(x, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_rtp(x);
#endif
}

static inline int __float2int_rz(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<int>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rint_rtz(x);
#endif
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_float2int_rd(float x) { return __float2int_rd(x); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_float2int_rn(float x) { return __float2int_rn(x); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_float2int_ru(float x) { return __float2int_ru(x); }

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_float2int_rz(float x) { return __float2int_rz(x); }

static inline unsigned int __float2uint_rd(float x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<unsigned int>(x, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_rtn(x);
#endif
}

static inline unsigned int __float2uint_rn(float x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<unsigned int>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_rte(x);
#endif
}

static inline unsigned int __float2uint_ru(float x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<unsigned int>(x, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_rtp(x);
#endif
}

static inline unsigned int __float2uint_rz(float x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<unsigned int>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Ruint_rtz(x);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_float2uint_rd(float x) {
  return __float2uint_rd(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_float2uint_rn(float x) {
  return __float2uint_rn(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_float2uint_ru(float x) {
  return __float2uint_ru(x);
}

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_float2uint_rz(float x) {
  return __float2uint_rz(x);
}

static inline long long int __float2ll_rd(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<long long int>(x, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_rtn(x);
#endif
}

static inline long long int __float2ll_rn(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<long long int>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_rte(x);
#endif
}

static inline long long int __float2ll_ru(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<long long int>(x, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_rtp(x);
#endif
}

static inline long long int __float2ll_rz(float x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<long long int>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToS_Rlong_rtz(x);
#endif
}

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_float2ll_rd(float x) { return __float2ll_rd(x); }

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_float2ll_rn(float x) { return __float2ll_rn(x); }

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_float2ll_ru(float x) { return __float2ll_ru(x); }

DEVICE_EXTERN_C_INLINE
long long int __devicelib_imf_float2ll_rz(float x) { return __float2ll_rz(x); }

static inline unsigned long long int __float2ull_rd(float x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<unsigned long long int>(x, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_rtn(x);
#endif
}

static inline unsigned long long int __float2ull_rn(float x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<unsigned long long int>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_rte(x);
#endif
}

static inline unsigned long long int __float2ull_ru(float x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<unsigned long long int>(x, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_rtp(x);
#endif
}

static inline unsigned long long int __float2ull_rz(float x) {
  if (x < 0)
    return 0;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __float2Tp_host<unsigned long long int>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertFToU_Rulong_rtz(x);
#endif
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_float2ull_rd(float x) {
  return __float2ull_rd(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_float2ull_rn(float x) {
  return __float2ull_rn(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_float2ull_ru(float x) {
  return __float2ull_ru(x);
}

DEVICE_EXTERN_C_INLINE
unsigned long long int __devicelib_imf_float2ull_rz(float x) {
  return __float2ull_rz(x);
}

DEVICE_EXTERN_C_INLINE
int __devicelib_imf_float_as_int(float x) { return __bit_cast<int, float>(x); }

DEVICE_EXTERN_C_INLINE
unsigned int __devicelib_imf_float_as_uint(float x) {
  return __bit_cast<unsigned int, float>(x);
}

static inline float __int2float_rd(int x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int, float>(x, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rfloat_rtn(x);
#endif
}

static inline float __int2float_rn(int x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int, float>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rfloat_rte(x);
#endif
}

static inline float __int2float_ru(int x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int, float>(x, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rfloat_rtp(x);
#endif
}

static inline float __int2float_rz(int x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int, float>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rfloat_rtz(x);
#endif
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_int2float_rd(int x) { return __int2float_rd(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_int2float_rn(int x) { return __int2float_rn(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_int2float_ru(int x) { return __int2float_ru(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_int2float_rz(int x) { return __int2float_rz(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_int_as_float(int x) { return __bit_cast<float, int>(x); }

static inline float __ll2float_rd(long long int x) {
  int64_t xi64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int64_t, float>(xi64, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rfloat_rtn(xi64);
#endif
}

static inline float __ll2float_rn(long long int x) {
  int64_t xi64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int64_t, float>(xi64, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rfloat_rte(xi64);
#endif
}

static inline float __ll2float_ru(long long int x) {
  int64_t xi64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int64_t, float>(xi64, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rfloat_rtp(xi64);
#endif
}

static inline float __ll2float_rz(long long int x) {
  int64_t xi64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<int64_t, float>(xi64, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertSToF_Rfloat_rtz(xi64);
#endif
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ll2float_rd(long long int x) { return __ll2float_rd(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ll2float_rn(long long int x) { return __ll2float_rn(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ll2float_ru(long long int x) { return __ll2float_ru(x); }

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ll2float_rz(long long int x) { return __ll2float_rz(x); }

static inline float __uint2float_rd(unsigned int x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<unsigned int, float>(x, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rfloat_rtn(x);
#endif
}

static inline float __uint2float_rn(unsigned int x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<unsigned int, float>(x, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rfloat_rte(x);
#endif
}

static inline float __uint2float_ru(unsigned int x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<unsigned int, float>(x, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rfloat_rtp(x);
#endif
}

static inline float __uint2float_rz(unsigned int x) {
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<unsigned int, float>(x, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rfloat_rtz(x);
#endif
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_uint2float_rd(unsigned int x) {
  return __uint2float_rd(x);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_uint2float_rn(unsigned int x) {
  return __uint2float_rn(x);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_uint2float_ru(unsigned int x) {
  return __uint2float_ru(x);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_uint2float_rz(unsigned int x) {
  return __uint2float_rz(x);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_uint_as_float(unsigned int x) {
  return __bit_cast<float, unsigned int>(x);
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ull2float_rd(unsigned long long int x) {
  uint64_t xui64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<uint64_t, float>(xui64, FE_DOWNWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rfloat_rtn(xui64);
#endif
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ull2float_rn(unsigned long long int x) {
  uint64_t xui64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<uint64_t, float>(xui64, FE_TONEAREST);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rfloat_rte(xui64);
#endif
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ull2float_ru(unsigned long long int x) {
  uint64_t xui64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<uint64_t, float>(xui64, FE_UPWARD);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rfloat_rtp(xui64);
#endif
}

DEVICE_EXTERN_C_INLINE
float __devicelib_imf_ull2float_rz(unsigned long long int x) {
  uint64_t xui64 = x;
#if defined(__LIBDEVICE_HOST_IMPL__)
  return __integral2FP_host<uint64_t, float>(xui64, FE_TOWARDZERO);
#elif defined(__SPIR__) || defined(__SPIRV__)
  return __spirv_ConvertUToF_Rfloat_rtz(xui64);
#endif
}

#endif // __LIBDEVICE_IMF_ENABLED__
