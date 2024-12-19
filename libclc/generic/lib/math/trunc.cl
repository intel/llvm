#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <libspirv/spirv.h>

#define __CLC_BUILTIN __spirv_ocl_trunc
#define __CLC_FUNCTION trunc
#include <math/unary_builtin.inc>
