
#include <clc/clc.h>
#include <spirv/spirv.h>

#include <clcmacro.h>

#define __CLC_BUILTIN __spirv_ocl_trunc
#define __CLC_FUNCTION trunc
#include <math/unary_builtin.inc>
