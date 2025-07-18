#include <clc/clcmacro.h>
#include <clc/utils.h>

#ifndef __CLC_BUILTIN
#define __CLC_BUILTIN __CLC_XCONCAT(__clc_, __CLC_FUNCTION)
#endif

#ifndef __CLC_BUILTIN_D
#define __CLC_BUILTIN_D __CLC_BUILTIN
#endif

#ifndef __CLC_BUILTIN_F
#define __CLC_BUILTIN_F __CLC_BUILTIN
#endif

#ifndef __CLC_BUILTIN_H
#define __CLC_BUILTIN_H __CLC_BUILTIN_F
#endif

#if (!defined(__HALF_ONLY) && !defined(__DOUBLE_ONLY))
_CLC_DEFINE_UNARY_BUILTIN_SCALARIZE(float, __CLC_FUNCTION, __CLC_BUILTIN_F, float)
#endif

#if (!defined(__HALF_ONLY) && !defined(__FLOAT_ONLY))
#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_UNARY_BUILTIN_SCALARIZE(double, __CLC_FUNCTION, __CLC_BUILTIN_D, double)

#endif
#endif // (!defined(__HALF_ONLY) && !defined(__FLOAT_ONLY))

#if (!defined(__FLOAT_ONLY) && !defined(__DOUBLE_ONLY))
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_SCALARIZE(half, __CLC_FUNCTION, __CLC_BUILTIN_H, half)

#endif
#endif // (!defined(__FLOAT_ONLY) && !defined(__DOUBLE_ONLY))
