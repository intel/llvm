
#include <clc/clc.h>
#include <spirv/spirv.h>

#include <clcmacro.h>

#define __CLC_BUILTIN __spirv_ocl_round
#define __CLC_FUNCTION round
#include <math/unary_builtin.inc>
