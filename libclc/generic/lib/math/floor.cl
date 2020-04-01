#include <spirv/spirv.h>
#include <clc/clc.h>
#include "../clcmacro.h"

#define __CLC_BUILTIN __spirv_ocl_floor
#define __CLC_FUNCTION floor
#include "unary_builtin.inc"
