C and C++ Standard libraries support
===================================

This extension enables a set of functions from the C and C++ standard
libraries, and allows to use them in SYCL device code.

Function declarations are taken from the standard headers (e.g. from
<assert.h> or <cassert>), and the corresponding header has to be
explicitly included in user code.

Implementation requires a special device library to be linked with a
SYCL program. The library should match the C or C++ standard library
used to compile the program.

List of supported functions from C standard library:
  - assert macro          (from <assert.h> or <cassert>)
  - logf, log             (from <math.h> or <cmath>)
  - expf, exp             (from <math.h> or <cmath>)
  - frexpf, frexp         (from <math.h> or <cmath>)
  - ldexpf, ldexp         (from <math.h> or <cmath>)
  - log10f, log10         (from <math.h> or <cmath>)
  - modff, modf           (from <math.h> or <cmath>)
  - exp2f, exp2           (from <math.h> or <cmath>)
  - expm1f, expm1         (from <math.h> or <cmath>)
  - ilogbf, ilogb         (from <math.h> or <cmath>)
  - log1pf, log1p         (from <math.h> or <cmath>)
  - log2f, log2           (from <math.h> or <cmath>)
  - logbf, logb           (from <math.h> or <cmath>)
  - sqrtf, sqrt           (from <math.h> or <cmath>)
  - cbrtf, cbrt           (from <math.h> or <cmath>)
  - hypotf, hypot         (from <math.h> or <cmath>)
  - erff, erf             (from <math.h> or <cmath>)
  - erfcf, erfc           (from <math.h> or <cmath>)
  - tgammaf, tgamma       (from <math.h> or <cmath>)
  - lgammaf, lgamma       (from <math.h> or <cmath>)
  - fmodf, fmod           (from <math.h> or <cmath>)
  - remainderf, remainder (from <math.h> or <cmath>)
  - remquof, remquo       (from <math.h> or <cmath>)
  - nextafterf, nextafter (from <math.h> or <cmath>)
  - fdimf, fdim           (from <math.h> or <cmath>)
  - fmaf, fma             (from <math.h> or <cmath>)
  - sinf, sin             (from <math.h> or <cmath>)
  - cosf, cos             (from <math.h> or <cmath>)
  - tanf, tan             (from <math.h> or <cmath>)
  - powf, pow             (from <math.h> or <cmath>)
  - acosf, acos           (from <math.h> or <cmath>)
  - asinf, asin           (from <math.h> or <cmath>)
  - atanf, atan           (from <math.h> or <cmath>)
  - atan2f, atan2         (from <math.h> or <cmath>)
  - coshf, cosh           (from <math.h> or <cmath>)
  - sinhf, sinh           (from <math.h> or <cmath>)
  - tanhf, tanh           (from <math.h> or <cmath>)
  - acoshf, acosh         (from <math.h> or <cmath>)
  - asinhf, asinh         (from <math.h> or <cmath>)
  - atanhf, atanh         (from <math.h> or <cmath>)
  - scalbnf, scalbn       (from <math.h> or <cmath>)
  - abs, labs, llabs      (from <stdlib.h> or <cstdlib>)
  - div, ldiv, lldiv      (from <stdlib.h> or <cstdlib>)
  - cimagf, cimag         (from <complex.h>)
  - crealf, creal         (from <complex.h>)
  - cargf, carg           (from <complex.h>)
  - cabsf, cabs           (from <complex.h>)
  - cprojf, cproj         (from <complex.h>)
  - cexpf, cexp           (from <complex.h>)
  - clogf, clog           (from <complex.h>)
  - cpowf, cpow           (from <complex.h>)
  - cpolarf, cpolar       (from <complex.h>)
  - csqrtf, csqrt         (from <complex.h>)
  - csinhf, csinh         (from <complex.h>)
  - ccoshf, ccosh         (from <complex.h>)
  - ctanhf, ctanh         (from <complex.h>)
  - csinf, csin           (from <complex.h>)
  - ccosf, ccos           (from <complex.h>)
  - ctanf, ctan           (from <complex.h>)
  - casinhf, casinh       (from <complex.h>)
  - cacoshf, cacosh       (from <complex.h>)
  - catanhf, catanh       (from <complex.h>)
  - casinf, casin         (from <complex.h>)
  - cacosf, cacos         (from <complex.h>)
  - catanf, catan         (from <complex.h>)
  - memcpy                (from <string.h>)
  - memset                (from <string.h>)
  - memcmp                (from <string.h>)

All functions are grouped into different device libraries based on
functionalities. C and C++ standard library groups functions and
classes by purpose(e.g. <math.h> for mathematical operations and
transformations) and device library infrastructure uses this as
a baseline.
NOTE: Only the GNU glibc, Microsoft C libraries are currently
supported. Not all functions from <math.h> are supported right now,
following math functions are not supported now:
 - lrintf, lrint
 - nexttowardf, nexttoward
 - nanf, nan

Device libraries can't support both single and double precision as some
underlying device may not support double precision.
'ldexpf' and 'frexpf' from MSVC <math.h> are implemented using corresponding
double precision version, they can be used only when double precision is
supported by underlying device.

All device libraries are linked by default. For example, no options need to be
added to use `assert` or math functions:
.. code:
   clang++ -fsycl main.cpp -o main.o

For Ahead-Of-Time compilation (AOT), the steps to use device libraries is
same, no options need to be added to use `assert` or math functions:
.. code:
   clang++ -fsycl -fsycl-targets=spir64_x86_64 main.cpp -o main.o

Example of usage
================

.. code: c++
   #include <assert.h>
   #include <CL/sycl.hpp>

   template <typename T, size_t N>
   void simple_vadd(const std::array<T, N>& VA, const std::array<T, N>& VB,
                    std::array<T, N>& VC) {
     // ...
     cl::sycl::range<1> numOfItems{N};
     cl::sycl::buffer<T, 1> bufferA(VA.data(), numOfItems);
     cl::sycl::buffer<T, 1> bufferB(VB.data(), numOfItems);
     cl::sycl::buffer<T, 1> bufferC(VC.data(), numOfItems);

     deviceQueue.submit([&](cl::sycl::handler& cgh) {
       auto accessorA = bufferA.template get_access<sycl_read>(cgh);
       auto accessorB = bufferB.template get_access<sycl_read>(cgh);
       auto accessorC = bufferC.template get_access<sycl_write>(cgh);

       cgh.parallel_for<class SimpleVadd<T>>(numOfItems,
       [=](cl::sycl::id<1> wiID) {
           accessorC[wiID] = accessorA[wiID] + accessorB[wiID];
           assert(accessorC[wiID] > 0 && "Invalid value");
       });
     });
     deviceQueue.wait_and_throw();
   }


.. code: c++
   #include <math.h>
   #include <CL/sycl.hpp>

   void device_sin_test() {
     cl::sycl::queue deviceQueue;
     cl::sycl::range<1> numOfItems{1};
     float  result_f = -1.f;
     double result_d = -1.d;
     {
       cl::sycl::buffer<float, 1> buffer1(&result_f, numOfItems);
       cl::sycl::buffer<double, 1> buffer2(&result_d, numOfItems);
       deviceQueue.submit([&](cl::sycl::handler &cgh) {
         auto res_access1 = buffer1.get_access<sycl_write>(cgh);
         auto res_access2 = buffer2.get_access<sycl_write>(cgh);
         cgh.single_task<class DeviceSin>([=]() {
           res_access1[0] = sinf(0.f);
           res_access2[0] = sin(0.0);
         });
       });
     }
     assert((result_f == 0.f) && (result_d == 0.0));
  }

Frontend
========

Once the system header is included, the corresponding functions can be
used in SYCL device code. This results in a handful of unresolved
functions in LLVM IR after clang:

.. code:
    ; Function Attrs: noreturn nounwind
    declare dso_local spir_func void @__assert_fail(i8 addrspace(4)*, i8 addrspace(4)*, i32, i8 addrspace(4)*)

    [...]
    cond.false:
      call spir_func void @__assert_fail([...])
      unreachable

The C and C++ specifications do not define names and signatures of the
functions from libc implementation that are used for a particular
function. For example, the `assert` macro:

  - in Glibc and musl libraries it expands to `__assert_fail`
  - in MSVC library it expands to `_wassert`
  - in newlib library it expands to `__assert_func`

This makes it difficult to handle all possible cases in device
compilers. In order to facilitate porting to new platforms, and to
avoid imposing a lot of boilerplate code in *every* device compiler,
wrapper libraries are provided with the SYCL compiler that "lower"
libc implementation-specific functions into a stable set of functions,
that can be later handled by a device compiler.

This `libsycl-crt.o` is one of these wrapper libraries: it provides
definitions for glibc specific library function, and these definitions
call the corresponding functions from `__devicelib_*` set of
functions.

For example, `__assert_fail` from IR above gets transformed into:
.. code:
    ; Function Attrs: noreturn nounwind
    declare dso_local spir_func void @__devicelib_assert_fail(i8 addrspace(4)*, i8 addrspace(4)*, i32, i8 addrspace(4)*)

    ; Function Attrs: noreturn nounwind
    define dso_local spir_func void @__assert_fail(i8 addrspace(4)*, i8 addrspace(4)*, i32, i8 addrspace(4)*) {
      call spir_func void @__devicelib_assert_fail([...])
    }

    [...]
    cond.false:
      call spir_func void @__assert_fail([...])
      unreachable

A single wrapper object provides function wrappers for *all* supported
library functions.

SPIR-V
======

Standard library functions are represented as external (import)
functions in SPIR-V:

.. code:
   8 Decorate 67 LinkageAttributes "__devicelib_assert_fail" Import
   ...
   2 Label 846
   8 FunctionCall 63 864 67 855 857 863 859
   1 Unreachable

Device compiler
===============

Device compiler is free to implement these `__devicelib_*` functions.
In order to indicate support for a particular set of functions,
underlying runtime have to support the corresponding OpenCL (PI)
extension. See ``../../internal-design/DeviceLibExtensions.rst`` for
a list of supported functions and corresponding extensions.

Fallback implementation
=======================

If a device compiler does not indicate "native" support for a
particular function, a fallback library is linked at JIT time by the
SYCL Runtime. This library is distributed with the SYCL Runtime and
resides in the same directory as the `libsycl.so` or `sycl.dll`.

A fallback library is implemented as a device-agnostic SPIR-V program,
and it is supposed to work for any device that supports SPIR-V.

Every set of functions is implemented in a separate fallback
library. For example, a fallback for `cl_intel_devicelib_cassert`
extension is provided as `libsycl-fallback-cassert.spv`

For AOT compilation, fallback libraries are provided as object files
(e.g. `libsycl-fallback-cassert.o`) which contain device code in LLVM
IR format. Device code in these object files is equivalent to device
code in the `*.spv` files. Those object files are located in compiler
package's 'lib/' folder.
