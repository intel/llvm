#include <clc/clc.h>
#include <spirv/spirv.h>

#include <clcmacro.h>

#define __CLC_BUILTIN __spirv_ocl_floor
#define __CLC_FUNCTION floor
#include <math/unary_builtin.inc>
