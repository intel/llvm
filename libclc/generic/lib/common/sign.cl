#include <clc/clc.h>
#include <spirv/spirv.h>

#include <clcmacro.h>

#define SIGN(TYPE, F) \
_CLC_DEF _CLC_OVERLOAD TYPE sign(TYPE x) { \
  return __spirv_ocl_sign(x); \
}

SIGN(float, f)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, sign, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

SIGN(double, )
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, sign, double)

#endif
