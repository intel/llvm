#include "../device_imf.hpp"

#ifdef __LIBDEVICE_IMF_ENABLED__

#define FLT_RTZ 0
#define FLT_RTE 1
#define FLT_RTU 2
#define FLT_RTD 3

DEVICE_EXTERN_C
float __attribute__((optnone)) __devicelib_imf_fadd_rz(float x, float y) {
  int r = __builtin_flt_rounds();
  __builtin_set_flt_rounds(FLT_RTZ);
  float z = x + y;
  __builtin_set_flt_rounds(r);
  return z;
}

DEVICE_EXTERN_C
float __attribute__((optnone)) __devicelib_imf_fadd_rn(float x, float y) {
  int r = __builtin_flt_rounds();
  __builtin_set_flt_rounds(FLT_RTE);
  float z = x + y;
  __builtin_set_flt_rounds(r);
  return z;
}

DEVICE_EXTERN_C
float __attribute__((optnone)) __devicelib_imf_fadd_ru(float x, float y) {
  int r = __builtin_flt_rounds();
  __builtin_set_flt_rounds(FLT_RTU);
  float z = x + y;
  __builtin_set_flt_rounds(r);
  return z;
}

DEVICE_EXTERN_C
float __attribute__((optnone)) __devicelib_imf_fadd_rd(float x, float y) {
  int r = __builtin_flt_rounds();
  __builtin_set_flt_rounds(FLT_RTD);
  float z = x + y;
  __builtin_set_flt_rounds(r);
  return z;
}

#endif
