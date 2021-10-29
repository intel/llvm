 #include <spirv/spirv.h>
 #include <clcmacro.h>

 #ifdef cl_khr_fp16
 #pragma OPENCL EXTENSION cl_khr_fp16 : enable
 #endif  

 double __ocml_log2_f64(double);
 float __ocml_log2_f32(float);
 half __ocml_log2_f16(half);
  #define __CLC_BUILTIN_H __CLC_XCONCAT(__CLC_BUILTIN, _f16)
 #define __CLC_FUNCTION __spirv_ocl_log2
 #define __CLC_BUILTIN __ocml_log2
 #define __CLC_BUILTIN_H __CLC_XCONCAT(__CLC_BUILTIN, _f16)
 #define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, _f32)
 #define __CLC_BUILTIN_D __CLC_XCONCAT(__CLC_BUILTIN, _f64)
// #include <math/unary_builtin.inc>

//#include <spirv/spirv.h>

#include "tables.h"
//#include <clcmacro.h>

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif // cl_khr_fp64

#define COMPILING_LOG2
#include "log_base.h"
#undef COMPILING_LOG2

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, __spirv_ocl_log2, float);

#ifdef cl_khr_fp64
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, __spirv_ocl_log2, double);
#endif // cl_khr_fp64

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
half __ocml_log2_f16(half);
#define __CLC_BUILTIN_H __CLC_XCONCAT(__CLC_BUILTIN, _f16)
_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __spirv_ocl_log2, half);
#endif // cl_khr_fp16
