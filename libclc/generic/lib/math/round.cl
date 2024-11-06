#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <spirv/spirv.h>

#define __CLC_BUILTIN __spirv_ocl_round
#define __CLC_FUNCTION round
#include <math/unary_builtin.inc>
