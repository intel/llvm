#include <clc/clc.h>
#include <libspirv/spirv.h>

#define __CLC_FUNC exp10
#define __CLC_SW_FUNC __spirv_ocl_exp10
#define __CLC_BODY <clc_sw_unary.inc>
#include <clc/math/gentype.inc>
#undef __CLC_SW_FUNC
