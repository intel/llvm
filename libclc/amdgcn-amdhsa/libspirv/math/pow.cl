 #include <spirv/spirv.h>
 #include <clcmacro.h>
 
 double __ocml_pow_f64(double, double);
 float __ocml_pow_f32(float, float);
 
 #define __CLC_FUNCTION __spirv_ocl_pow
 #define __CLC_BUILTIN __ocml_pow
 #define __CLC_BUILTIN_F __CLC_XCONCAT(__CLC_BUILTIN, _f32)
 #define __CLC_BUILTIN_D __CLC_XCONCAT(__CLC_BUILTIN, _f64)
 #include <math/binary_builtin.inc> 
