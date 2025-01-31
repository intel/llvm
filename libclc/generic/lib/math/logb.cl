#include <clc/clc.h>
#include <clc/clcmacro.h>
<<<<<<< HEAD
#include <libspirv/spirv.h>
=======
#include <clc/math/math.h>
>>>>>>> 78b5bb702fe97fe85f66d72598d0dfa7c49fe001

_CLC_OVERLOAD _CLC_DEF float logb(float x) {
    return __spirv_ocl_logb(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, logb, float);

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double logb(double x) {
    return __spirv_ocl_logb(x);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, logb, double)
#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_UNARY_BUILTIN_FP16(logb)

#endif
