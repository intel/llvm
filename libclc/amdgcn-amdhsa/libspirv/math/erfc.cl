 #include <spirv/spirv.h>
 #include <clcmacro.h>
 
 double __ocml_erfc_f64(double);
 float __ocml_erfc_f32(float);
 
 #define __CLC_FUNCTION __spirv_ocl_erfc
 #define __CLC_BUILTIN __ocml_erfc
 #define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, _f32)
 #define __CLC_BUILTIN_D __CLC_XCONCAT(__CLC_BUILTIN, _f64)
 #include <math/unary_builtin.inc>
