
#include <clc/clc.h>
#include <spirv/spirv.h>

#include "../../libspirv/math/tables.h"
#include <clcmacro.h>

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif // cl_khr_fp64

_CLC_OVERLOAD _CLC_DEF long int lround(float x) {
    return __spirv_ocl_lround(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long int, lround, float);

#ifdef cl_khr_fp64
_CLC_OVERLOAD _CLC_DEF long int lround(double x) {
    return __spirv_ocl_lround(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, long int, lround, double);
#endif // cl_khr_fp64
