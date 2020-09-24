#include <clc/clc.h>
#include <spirv/spirv.h>

#include <clcmacro.h>

#define __CLC_BUILTIN __spirv_ocl_fabs
#define __CLC_FUNCTION fabs
#include <math/unary_builtin.inc>
