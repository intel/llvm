#include <clc/clc.h>
#include <math/clc_remquo.h>

#define __CLC_BODY <remquo.inc>
#define __CLC_ADDRESS_SPACE global
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_BODY <remquo.inc>
#define __CLC_ADDRESS_SPACE local
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#define __CLC_BODY <remquo.inc>
#define __CLC_ADDRESS_SPACE private
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE

#if __OPENCL_C_VERSION__ == CL_VERSION_2_0 ||                                  \
    (__OPENCL_C_VERSION__ >= CL_VERSION_3_0 &&                                 \
     defined(__opencl_c_generic_address_space))
#define __CLC_BODY <remquo.inc>
#define __CLC_ADDRESS_SPACE generic
#include <clc/math/gentype.inc>
#undef __CLC_ADDRESS_SPACE
#endif
