#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <spirv/spirv.h>

#define __CLC_BUILTIN __spirv_ocl_native_sin
#define __CLC_FUNCTION native_sin
#define __CLC_BODY <native_builtin.inc>
#define __FLOAT_ONLY
#include <clc/math/gentype.inc>
