#include <clc/clc.h>
#include <spirv/spirv.h>

#include <clcmacro.h>

#define __CLC_BUILTIN __spirv_ocl_ceil
#define __CLC_FUNCTION ceil
#include <math/unary_builtin.inc>
