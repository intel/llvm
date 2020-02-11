C and C++ Standard libraries support
===================================

This extension enables a set of functions from the C and C++ standard
libraries, and allows to use them in SYCL device code.

Function declarations are taken from the standard headers (e.g. from
<assert.h> or <cassert>), and the corresponding header has to be
explicitly included in user code.

Implementation requires a special device library to be linked with a
SYCL program. The library should match the C or C++ standard library
used to compile the program:

For example, on Linux with GNU glibc:
.. code:
   clang++ -fsycl -c main.cpp -o main.o
   clang++ -fsycl main.o $(SYCL_INSTALL)/lib/libsycl-glibc.o -o a.out

or, in case of Windows:
.. code:
   clang++ -fsycl -c main.cpp -o main.obj
   clang++ -fsycl main.obj %SYCL_INSTALL%/lib/libsycl-msvc.o -o a.exe

List of supported functions from C standard library:
  - assert macro (from <assert.h> or <cassert>)

NOTE: only the GNU glibc and Microsoft C libraries are currently
supported.

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

.. code:
   clang++ -fsycl -c main.cpp -o main.o
   clang++ -fsycl main.o $(SYCL_INSTALL)/lib/libsycl-glibc.o -o a.out

This `libsycl-glibc.o` is one of these wrapper libraries: it provides
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
library functions. Every supported C library implementation (MSVC or
glibc) has its own wrapper library object:

  - libsycl-glibc.o
  - libsycl-msvc.o

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
extension. See ``DeviceLibExtensions.rst`` for a list of supported
functions and corresponding extensions.

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

NOTE that AOT compilation is not yet supported. Driver will have to
check for extension support and link the corresponding SPIR-V fallback
implementation, but this is not implemented yet.
