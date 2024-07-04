#include <clc/clc.h>
#include <spirv/spirv.h>

#include <clcmacro.h>

// Use __spirv_ocl_round for rounding and cast the result to long
_CLC_OVERLOAD _CLC_DEF long lround(float x) {
    return __spirv_ocl_lround(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, lround, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF long lround(double x) {
    return __spirv_ocl_lround(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, lround, double);

#endif // cl_khr_fp64

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF long lround(half x) {
    return __spirv_ocl_lround(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long, lround, half);

#endif // cl_khr_fp16
