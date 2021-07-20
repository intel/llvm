Device library extensions
===================================

Device compiler that indicates support for a particular extension is
supposed to support *all* the corresponding functions.

cl_intel_devicelib_cassert
==========================

.. code:
   void __devicelib_assert_fail(__generic const char *expr,
                                __generic const char *file,
                                int32_t line,
                                __generic const char *func,
                                size_t gid0, size_t gid1, size_t gid2,
                                size_t lid0, size_t lid1, size_t lid2);
Semantic:
the function is called when an assertion expression `expr` is false,
and it indicates that a program does not execute as expected.
The function should print a message containing the information
provided in the arguments. In addition to that, the function is free
to terminate the current kernel invocation.

Arguments:

  - `expr` is a string representation of the assertion condition
  - `file` and `line` are the source code location of the assertion
  - `func` (optional, may be NULL)  name of the function containing the assertion
  - `gidX` current work-item global id
  - `lidX` current work-item local id

Example of a message:
.. code:
   foo.cpp:42: void foo(int): global id: [0,0,0], local id: [0,0,0] Assertion `buf[wiID] == 0 && "Invalid value"` failed.

See also: assert_extension_.
.. _assert_extension: ../Assert/SYCL_ONEAPI_ASSERT.asciidoc)

cl_intel_devicelib_math
==========================

.. code:
   int    __devicelib_abs(int x);
   int    __devicelib_labs(long int x);
   int    __devicelib_llabs(long long int x);
   int    __devicelib_div(int x, int y);
   int    __devicelib_ldiv(long int x, long int y);
   int    __devicelib_lldiv(long long int x, long long int y);
   float  __devicelib_scalbnf(float x, int n);
   float  __devicelib_logf(float x);
   float  __devicelib_sinf(float x);
   float  __devicelib_cosf(float x);
   float  __devicelib_tanf(float x);
   float  __devicelib_acosf(float x);
   float  __devicelib_powf(float x, float y);
   float  __devicelib_sqrtf(float x);
   float  __devicelib_cbrtf(float x);
   float  __devicelib_hypotf(float x, float y);
   float  __devicelib_erff(float x);
   float  __devicelib_erfcf(float x);
   float  __devicelib_tgammaf(float x);
   float  __devicelib_lgammaf(float x);
   float  __devicelib_fmodf(float x, float y);
   float  __devicelib_remainderf(float x, float y);
   float  __devicelib_remquof(float x, float y, int *q);
   float  __devicelib_nextafterf(float x, float y);
   float  __devicelib_fdimf(float x, float y);
   float  __devicelib_fmaf(float x, float y, float z);
   float  __devicelib_asinf(float x);
   float  __devicelib_atanf(float x);
   float  __devicelib_atan2f(float x, float y);
   float  __devicelib_coshf(float x);
   float  __devicelib_sinhf(float x);
   float  __devicelib_tanhf(float x);
   float  __devicelib_acoshf(float x);
   float  __devicelib_asinhf(float x);
   float  __devicelib_atanhf(float x);
   float  __devicelib_frexpf(float x, int *exp);
   float  __devicelib_ldexpf(float x, int exp);
   float  __devicelib_log10f(float x);
   float  __devicelib_modff(float x, float *intpart);
   float  __devicelib_expf(float x);
   float  __devicelib_exp2f(float x);
   float  __devicelib_expm1f(float x);
   int    __devicelib_ilogbf(float x);
   float  __devicelib_log1pf(float x);
   float  __devicelib_log2f(float x);
   float  __devicelib_logbf(float x);

Semantic:
Those __devicelib_* functions perform the same operation as the corresponding C math
library functions for single precision. These functions do not support errno, and on
some devices floating-point exceptions may not be raised.

Arguments:
Those __devicelib_* functions have the same argument type and return type as corresponding
math functions from <math.h>, please refer to ISO/IEC 14882:2011 for details.

cl_intel_devicelib_math_fp64
==========================

.. code:
   double __devicelib_scalbn(double x, int exp);
   double __devicelib_log(double x);
   double __devicelib_sin(double x);
   double __devicelib_cos(double x);
   double __devicelib_tan(double x);
   double __devicelib_acos(double x);
   double __devicelib_pow(double x, double y);
   double __devicelib_sqrt(double x);
   double __devicelib_cbrt(double x);
   double __devicelib_hypot(double x, double y);
   double __devicelib_erf(double x);
   double __devicelib_erfc(double x);
   double __devicelib_tgamma(double x);
   double __devicelib_lgamma(double x);
   double __devicelib_fmod(double x, double y);
   double __devicelib_remainder(double x, double y);
   double __devicelib_remquo(double x, double y, int *q);
   double __devicelib_nextafter(double x, double y);
   double __devicelib_fdim(double x, double y);
   double __devicelib_fma(double x, double y, double z);
   double __devicelib_asin(double x);
   double __devicelib_atan(double x);
   double __devicelib_atan2(double x, double y);
   double __devicelib_cosh(double x);
   double __devicelib_sinh(double x);
   double __devicelib_tanh(double x);
   double __devicelib_acosh(double x);
   double __devicelib_asinh(double x);
   double __devicelib_atanh(double x);
   double __devicelib_frexp(double x, int *exp);
   double __devicelib_ldexp(double x, int exp);
   double __devicelib_log10(double x);
   double __devicelib_modf(double x, double *intpart);
   double __devicelib_exp(double x);
   double __devicelib_exp2(double x);
   double __devicelib_expm1(double x);
   int    __devicelib_ilogb(double x);
   double __devicelib_log1p(double x);
   double __devicelib_log2(double x);
   double __devicelib_logb(double x);

Semantic:
Those __devicelib_* functions perform the same operation as the corresponding C math
library functions for double precision. These functions do not support errno, and on
some devices floating-point exceptions may not be raised.

Arguments:
Those __devicelib_* functions have the same argument type and return type as corresponding
math functions from <math.h>, please refer to ISO/IEC 14882:2011 for details.

cl_intel_devicelib_complex
==========================

.. code:
   float  __devicelib_cimagf(float __complex__ z);
   float  __devicelib_crealf(float __complex__ z);
   float  __devicelib_cargf(float __complex__ z);
   float  __devicelib_cabsf(float __complex__ z);
   float  __complex__ __devicelib_cprojf(float __complex__ z);
   float  __complex__ __devicelib_cexpf(float __complex__ z);
   float  __complex__ __devicelib_clogf(float __complex__ z);
   float  __complex__ __devicelib_cpowf(float __complex__ x, float __complex__ y);
   float  __complex__ __devicelib_cpolarf(float x, float y);
   float  __complex__ __devicelib_csqrtf(float __complex__ z);
   float  __complex__ __devicelib_csinhf(float __complex__ z);
   float  __complex__ __devicelib_ccoshf(float __complex__ z);
   float  __complex__ __devicelib_ctanhf(float __complex__ z);
   float  __complex__ __devicelib_csinf(float __complex__ z);
   float  __complex__ __devicelib_ccosf(float __complex__ z);
   float  __complex__ __devicelib_ctanf(float __complex__ z);
   float  __complex__ __devicelib_cacosf(float __complex__ z);
   float  __complex__ __devicelib_casinhf(float __complex__ z);
   float  __complex__ __devicelib_casinf(float __complex__ z);
   float  __complex__ __devicelib_cacoshf(float __complex__ z);
   float  __complex__ __devicelib_catanhf(float __complex__ z);
   float  __complex__ __devicelib_catanf(float __complex__ z);
   float  __complex__ __devicelib___mulsc3(float a, float b, float c, float d);
   float  __complex__ __devicelib___divsc3(float a, float b, float c, float d);

Semantic:
Those __devicelib_* functions perform the same operation as the corresponding C math
library functions for single precision. These functions do not support errno, and on
some devices floating-point exceptions may not be raised.

Arguments:
Those __devicelib_* functions have the same argument type and return type as corresponding
complex math functions from <complex.h>, please refer to ISO/IEC 14882:2011 for details. The
"float __complex__" type is C99 complex type and it is an alias to "struct {float, float}"
in LLVM IR and SPIR-V.

cl_intel_devicelib_complex_fp64
==========================

.. code:
   double __devicelib_cimag(double __complex__ z);
   double __devicelib_creal(double __complex__ z);
   double __devicelib_carg(double __complex__ z);
   double __devicelib_cabs(double __complex__ z);
   double __complex__ __devicelib_cproj(double __complex__ z);
   double __complex__ __devicelib_cexp(double __complex__ z);
   double __complex__ __devicelib_clog(double __complex__ z);
   double __complex__ __devicelib_cpow(double __complex__ x, double __complex__ y);
   double __complex__ __devicelib_cpolar(double x, double y);
   double __complex__ __devicelib_csqrt(double __complex__ z);
   double __complex__ __devicelib_csinh(double __complex__ z);
   double __complex__ __devicelib_ccosh(double __complex__ z);
   double __complex__ __devicelib_ctanh(double __complex__ z);
   double __complex__ __devicelib_csin(double __complex__ z);
   double __complex__ __devicelib_ccos(double __complex__ z);
   double __complex__ __devicelib_ctan(double __complex__ z);
   double __complex__ __devicelib_cacos(double __complex__ z);
   double __complex__ __devicelib_casinh(double __complex__ z);
   double __complex__ __devicelib_casin(double __complex__ z);
   double __complex__ __devicelib_cacosh(double __complex__ z);
   double __complex__ __devicelib_catanh(double __complex__ z);
   double __complex__ __devicelib_catan(double __complex__ z);
   double __complex__ __devicelib___muldc3(double a, double b, double c, double d);
   double __complex__ __devicelib___divdc3(double a, double b, double c, double d);

Semantic:
Those __devicelib_* functions perform the same operation as the corresponding C math
library functions for double precision. These functions do not support errno, and on
some devices floating-point exceptions may not be raised.

Arguments:
Those __devicelib_* functions have the same argument type and return type as corresponding
complex math functions from <complex.h>, please refer to ISO/IEC 14882:2011 for details. The
"double __complex__" type is C99 complex type and it is an alias to "struct {double, double}"
in LLVM IR and SPIR-V.

cl_intel_devicelib_cstring
==========================

.. code:
   void *__devicelib_memcpy(void *dest, const void *src, size_t n);
   void *__devicelib_memset(void *dest, int c, size_t n);
   int __devicelib_memcmp(const void *s1, const void *s2, size_t n);

Semantic:
Those __devicelib_* functions perform the same operation as the corresponding C string
library functions.

Arguments:
Those __devicelib_* functions have the same argument type and return type as corresponding
string functions from <string.h>, please refer to ISO/IEC 14882:2011 for details.
