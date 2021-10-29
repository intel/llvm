 #include <spirv/spirv.h>
 #include <clcmacro.h>
 
 double __ocml_log_f64(double);
 float __ocml_log_f32(float);
 
 #define __CLC_FUNCTION __spirv_ocl_log
 #define __CLC_BUILTIN __ocml_log
 #define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, _f32)
 #define __CLC_BUILTIN_D __CLC_XCONCAT(__CLC_BUILTIN, _f64)
 #include <math/unary_builtin.inc>
