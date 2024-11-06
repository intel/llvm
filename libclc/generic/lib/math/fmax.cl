#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <spirv/spirv.h>

_CLC_DEFINE_BINARY_BUILTIN(float, fmax, __spirv_ocl_fmax, float, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_BINARY_BUILTIN(double, fmax, __spirv_ocl_fmax, double, double);

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEF _CLC_OVERLOAD half fmax(half x, half y)
{
   return __spirv_ocl_fmax(x, y);
}
_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, fmax, half, half)

#endif

#define __CLC_BODY <fmax.inc>
#include <clc/math/gentype.inc>
