// REQUIRES: windows
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %{build} -DBUILD_LIB_A %shared_lib -o %t.dir/lib_a.dll
// RUN: %{build} -DBUILD_LIB_B %shared_lib -o %t.dir/lib_b.dll
// RUN: %{build} -DBUILD_MAIN '-DLIB_DIR="%/t.dir"' -o %t.out
// RUN: %{run} %t.out
//
// Test verifies that free function kernels with same name but DIFFERENT
// SIGNATURES work correctly on Windows. No special linker flags needed
// because different signatures = different function types = no interposition.

#define LIB_A "lib_a.dll"
#define LIB_B "lib_b.dll"
#include "../kernel_name_collision_free_function_diff_sig.inc"
