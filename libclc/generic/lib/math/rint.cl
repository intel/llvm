#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <spirv/spirv.h>

#define __CLC_BUILTIN __spirv_ocl_rint
#define __CLC_FUNCTION rint
#include <math/unary_builtin.inc>
