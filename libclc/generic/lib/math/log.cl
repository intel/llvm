#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <spirv/spirv.h>

/*
 *log(x) = log2(x) * (1/log2(e))
 */

_CLC_OVERLOAD _CLC_DEF float log(float x)
{
    return __spirv_ocl_log(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, log, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double log(double x)
{
    return __spirv_ocl_log(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, log, double);

#endif // cl_khr_fp64
