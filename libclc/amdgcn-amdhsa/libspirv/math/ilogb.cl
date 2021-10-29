 #include <spirv/spirv.h>
 #include <clcmacro.h>
 
 int __ocml_ilogb_f64(double);
 int __ocml_ilogb_f32(float);
 
 #define __CLC_FUNCTION __spirv_ocl_ilogb
 #define __CLC_BUILTIN __ocml_ilogb
 #define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, _f32)
 #define __CLC_BUILTIN_D __CLC_XCONCAT(__CLC_BUILTIN, _f64)
 #include <math/unary_builtin.inc>
