// REQUIRES: linux
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %{build} -DBUILD_SMALL_LIB -fPIC -shared -o %t.dir/small.so
// RUN: %{build} -DBUILD_LARGE_LIB -fPIC -shared -o %t.dir/large.so
// RUN: %{build} -DBUILD_MAIN '-DLIB_DIR="%t.dir"' -ldl -o %t.out
// RUN: %{run} %t.out
//
// Test verifies that RTLD_LOCAL loading prevents symbol interposition,
// ensuring each DSO's kernel is correctly dispatched even with forward-declared
// kernel name types.

#define DLOPEN_FLAGS (RTLD_NOW | RTLD_LOCAL)
#define KERNEL_CLASS_DECL class KernelFunctor;
#define LIB_SMALL "small.so"
#define LIB_LARGE "large.so"
#include "../kernel_name_collision.inc"
