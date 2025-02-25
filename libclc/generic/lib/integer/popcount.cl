#include <clc/clc.h>
#include <libspirv/spirv.h>

#define __CLC_FUNC popcount
#define __CLC_IMPL_FUNC __spirv_ocl_popcount

#define __CLC_BODY "../clc_unary.inc"
#include <clc/integer/gentype.inc>
