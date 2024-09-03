#include "utils.h"
#include <clcmacro.h>

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

// Define the lround function for float type
#define _CLC_DEFINE_LROUND_BUILTIN(FUNC, BUILTIN, TYPE) \
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN long FUNC(TYPE x) { \
    return (long)BUILTIN(x); \
}

#define _CLC_DEFINE_LROUND_VECTOR_BUILTIN(FUNC, BUILTIN, VTYPE, RTYPE) \
_CLC_OVERLOAD _CLC_DECL _CLC_CONSTFN RTYPE FUNC(VTYPE x) { \
    return (RTYPE)BUILTIN(x); \
}

#define __CLC_FUNCTION lround

_CLC_DEFINE_LROUND_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, float)

#ifndef __FLOAT_ONLY

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
_CLC_DEFINE_LROUND_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, double)
#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
_CLC_DEFINE_LROUND_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, half)
#endif

#endif // !__FLOAT_ONLY

// Define lround for vector types of float
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec2_float, __clc_vec2_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec3_float, __clc_vec3_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec4_float, __clc_vec4_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec8_float, __clc_vec8_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec16_float, __clc_vec16_long)

#ifdef cl_khr_fp64
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec2_double, __clc_vec2_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec3_double, __clc_vec3_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec4_double, __clc_vec4_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec8_double, __clc_vec8_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec16_double, __clc_vec16_long)
#endif

#ifdef cl_khr_fp16
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec2_half, __clc_vec2_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec3_half, __clc_vec3_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec4_half, __clc_vec4_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec8_half, __clc_vec8_long)
_CLC_DEFINE_LROUND_VECTOR_BUILTIN(__spirv_ocl_lround, __spirv_ocl_rint, __clc_vec16_half, __clc_vec16_long)
#endif

#undef __CLC_FUNCTION
#undef __CLC_BUILTIN
#undef __CLC_BUILTIN_D
#undef __CLC_BUILTIN_F
#undef __CLC_BUILTIN_H
