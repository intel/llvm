
#include <clc/clc.h>
#include <spirv/spirv.h>

#include <clcmacro.h>

#define __CLC_BUILTIN __spirv_ocl_native_sqrt
#define __CLC_FUNCTION native_sqrt
#define __CLC_BODY <native_builtin.inc>
#define __FLOAT_ONLY
#include <clc/math/gentype.inc>
