#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_fabs.h>
#include <libspirv/spirv.h>

#define __CLC_BUILTIN __spirv_ocl_fabs
#define __CLC_FUNCTION fabs
#include <math/unary_builtin.inc>
