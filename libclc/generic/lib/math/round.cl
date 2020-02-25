
#include <spirv/spirv.h>
#include <clc/clc.h>
#include "../clcmacro.h"

#define __CLC_BUILTIN __spirv_ocl_round
#define __CLC_FUNCTION round
#include "unary_builtin.inc"
