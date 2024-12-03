#include "math.h"
#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <spirv/spirv.h>

_CLC_OVERLOAD _CLC_DEF float logb(float x) {
    return __spirv_ocl_logb(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, logb, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double logb(double x) {
    return __spirv_ocl_logb(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, logb, double)
#endif
