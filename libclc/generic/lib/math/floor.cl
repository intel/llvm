#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_floor.h>
#include <libspirv/spirv.h>

#define __CLC_BUILTIN __spirv_ocl_floor
#define __CLC_FUNCTION floor
#include <math/unary_builtin.inc>
