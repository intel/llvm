#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <spirv/spirv.h>

/* Refer to the exp routine for the underlying algorithm */

_CLC_OVERLOAD _CLC_DEF float expm1(float x) {
    return __spirv_ocl_expm1(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, expm1, float)

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double expm1(double x) {
    return __spirv_ocl_expm1(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, expm1, double)

#endif
