// REQUIRES: linux
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %{build} -DBUILD_LIB_A -fPIC -shared -o %t.dir/lib_a.so
// RUN: %{build} -DBUILD_LIB_B -fPIC -shared -o %t.dir/lib_b.so
// RUN: %{build} -DBUILD_MAIN '-DLIB_DIR="%t.dir"' -ldl -o %t.out
// RUN: %{run} %t.out
//
// Test verifies that free function kernels with same name but DIFFERENT
// SIGNATURES work correctly even with RTLD_GLOBAL. No -Wl,-Bsymbolic needed
// because different signatures = different function types = no interposition.

#define DLOPEN_FLAGS (RTLD_NOW | RTLD_GLOBAL)
#define LIB_A "lib_a.so"
#define LIB_B "lib_b.so"
#include "../kernel_name_collision_free_function_diff_sig.inc"
