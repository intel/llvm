#include <clc/clc.h>
<<<<<<< HEAD
#include <libspirv/spirv.h>
=======
#include <clc/clcmacro.h>
#include <clc/math/clc_fma.h>
#include <clc/math/math.h>

_CLC_DEFINE_TERNARY_BUILTIN(float, fma, __clc_fma, float, float, float)
>>>>>>> e7ad07ffb846a9812d9567b8d4b680045dce5b28

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEFINE_TERNARY_BUILTIN(double, fma, __clc_fma, double, double, double)

#endif

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_DEFINE_TERNARY_BUILTIN(half, fma, __clc_fma, half, half, half)

#endif
