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

#define __CLC_SCALAR
#define __CLC_MIN_VECSIZE 1

#if (!defined(__CLC_HALF_ONLY) && !defined(__CLC_DOUBLE_ONLY))

#define __CLC_IMPL_FUNCTION __CLC_BUILTIN_F
#define __CLC_GENTYPE float
#include <clc/shared/unary_def_scalarize.inc>
#undef __CLC_GENTYPE
#undef __CLC_IMPL_FUNCTION

#endif // (!defined(__CLC_HALF_ONLY) && !defined(__CLC_DOUBLE_ONLY))

#if (!defined(__CLC_HALF_ONLY) && !defined(__CLC_FLOAT_ONLY))
#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define __CLC_IMPL_FUNCTION __CLC_BUILTIN_D
#define __CLC_GENTYPE double
#include <clc/shared/unary_def_scalarize.inc>
#undef __CLC_GENTYPE
#undef __CLC_IMPL_FUNCTION

#endif
#endif // (!defined(__CLC_HALF_ONLY) && !defined(__CLC_FLOAT_ONLY))

#if (!defined(__CLC_FLOAT_ONLY) && !defined(__CLC_DOUBLE_ONLY))
#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define __CLC_IMPL_FUNCTION __CLC_BUILTIN_H
#define __CLC_GENTYPE half
#include <clc/shared/unary_def_scalarize.inc>
#undef __CLC_GENTYPE
#undef __CLC_IMPL_FUNCTION

#endif
#endif // (!defined(__CLC_FLOAT_ONLY) && !defined(__CLC_DOUBLE_ONLY))

#undef __CLC_SCALAR
#undef FUNCTION
#undef __CLC_MIN_VECSIZE
