// #include <spirv/spirv.h>
 
// double __clc_remainder(double, double);
// float __clc_remainderf(float, float);
//  #ifdef cl_khr_fp16
// #pragma OPENCL EXTENSION cl_khr_fp16 : enable
// half __clc_remainderf(half, half);
// #endif 
// #define __CLC_FUNCTION __spirv_ocl_remainder
// #define __CLC_BUILTIN __clc_remainder
// #define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, f)
// #include <math/binary_builtin.inc>

#include <spirv/spirv.h>

#include <math/clc_remainder.h>

#define __CLC_FUNC __spirv_ocl_remainder
#define __CLC_SW_FUNC __clc_remainder
#define __CLC_BODY <clc_sw_binary.inc>
#include <clc/math/gentype.inc>