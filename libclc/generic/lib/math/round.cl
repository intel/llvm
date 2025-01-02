#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <libspirv/spirv.h>

#define __CLC_BUILTIN __spirv_ocl_round
#define __CLC_FUNCTION round
#include <clc/math/unary_builtin.inc>
