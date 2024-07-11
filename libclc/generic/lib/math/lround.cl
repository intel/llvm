
#include <clc/clc.h>
#include <spirv/spirv.h>

#include <clcmacro.h>

#define __CLC_BUILTIN __spirv_ocl_lround
#define __CLC_FUNCTION lround
#include <math/lround_builtin.inc>
