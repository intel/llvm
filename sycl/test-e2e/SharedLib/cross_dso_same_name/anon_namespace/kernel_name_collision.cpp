// REQUIRES: linux
// UNSUPPORTED: native_cpu
// UNSUPPORTED-INTENDED: native_cpu compiles kernels as native function symbols
// that get interposed by the dynamic linker with RTLD_GLOBAL.
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %{build} -DBUILD_SMALL_LIB -fPIC -shared -o %t.dir/small.so
// RUN: %{build} -DBUILD_LARGE_LIB -fPIC -shared -o %t.dir/large.so
// RUN: %{build} -DBUILD_MAIN '-DLIB_DIR="%t.dir"' -ldl -o %t.out
// RUN: %{run} %t.out
//
// Test verifies that RTLD_GLOBAL loading with anonymous namespace kernel types
// works correctly. Direct kernel submissions using local lambda/functor types
// with internal linkage work even with RTLD_GLOBAL because each DSO has its
// own type and the runtime disambiguates by OS module handle.

#define DLOPEN_FLAGS (RTLD_NOW | RTLD_GLOBAL)
#define KERNEL_CLASS_DECL                                                      \
  namespace {                                                                  \
  class KernelFunctor;                                                         \
  }
#define LIB_SMALL "small.so"
#define LIB_LARGE "large.so"
#include "../kernel_name_collision.inc"
