#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_ceil.h>
#include <libspirv/spirv.h>

#define __CLC_BUILTIN __spirv_ocl_ceil
#define __CLC_FUNCTION ceil
#include <math/unary_builtin.inc>
