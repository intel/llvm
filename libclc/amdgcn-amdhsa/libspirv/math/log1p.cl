 #include <spirv/spirv.h>
 #include <clcmacro.h>
 
 double __ocml_log1p_f64(double);
 float __ocml_log1p_f32(float);
 
 #define __CLC_FUNCTION __spirv_ocl_log1p
 #define __CLC_BUILTIN __ocml_log1p
 #define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, _f32)
 #define __CLC_BUILTIN_D __CLC_XCONCAT(__CLC_BUILTIN, _f64)
 #include <math/unary_builtin.inc>
