#include <clc/clcmacro.h>

_CLC_OVERLOAD _CLC_DEF float __clc_degrees(float radians) {
  // 180/pi = ~57.29577951308232087685 or 0x1.ca5dc1a63c1f8p+5 or 0x1.ca5dc2p+5F
  return 0x1.ca5dc2p+5F * radians;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __clc_degrees, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __clc_degrees(double radians) {
  // 180/pi = ~57.29577951308232087685 or 0x1.ca5dc1a63c1f8p+5 or 0x1.ca5dc2p+5F
  return 0x1.ca5dc1a63c1f8p+5 * radians;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __clc_degrees, double);

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __clc_degrees(half radians) {
  return __clc_degrees((float)radians);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __clc_degrees, half);

#endif
