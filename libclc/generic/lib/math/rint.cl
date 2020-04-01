
#include <spirv/spirv.h>
#include <clc/clc.h>
#include "../clcmacro.h"

#define __CLC_BUILTIN __spirv_ocl_rint
#define __CLC_FUNCTION rint
#include "unary_builtin.inc"
