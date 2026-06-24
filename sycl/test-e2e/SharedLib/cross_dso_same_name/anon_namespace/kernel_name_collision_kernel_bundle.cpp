// REQUIRES: linux
// UNSUPPORTED: native_cpu
// UNSUPPORTED-INTENDED: native_cpu compiles kernels as native function symbols
// that get interposed by the dynamic linker with RTLD_GLOBAL.
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %{build} -DBUILD_LIB_A -fPIC -shared -o %t.dir/lib_a.so
// RUN: %{build} -DBUILD_LIB_B -fPIC -shared -o %t.dir/lib_b.so
// RUN: %{build} -DBUILD_MAIN '-DLIB_DIR="%t.dir"' -ldl -o %t.out
// RUN: %{run} %t.out
//
// Test verifies that kernel_bundle APIs work correctly with RTLD_GLOBAL when
// using anonymous namespace kernel types (internal linkage). Each DSO has its
// own KernelFunctor type, preventing symbol interposition.

#define DLOPEN_FLAGS (RTLD_NOW | RTLD_GLOBAL)
#define KERNEL_CLASS_DECL                                                      \
  namespace {                                                                  \
  class KernelFunctor;                                                         \
  }
#define LIB_A "lib_a.so"
#define LIB_B "lib_b.so"
#include "../kernel_name_collision_kernel_bundle.inc"
