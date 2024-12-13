#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <libspirv/spirv.h>

_CLC_DEFINE_BINARY_BUILTIN(float, fmin, __spirv_ocl_fmin, float, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, fmin, __spirv_ocl_fmin, double, double);

#endif
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half fmin(half x, half y)
{
   return __spirv_ocl_fmin(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, fmin, half, half)

#endif

#define __CLC_BODY <fmin.inc>
#include <clc/math/gentype.inc>
