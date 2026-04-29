// REQUIRES: windows
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %{build} -DBUILD_LIB_A %shared_lib -o %t.dir/lib_a.dll
// RUN: %{build} -DBUILD_LIB_B %shared_lib -o %t.dir/lib_b.dll
// RUN: %{build} -DBUILD_MAIN '-DLIB_DIR="%/t.dir"' -o %t.out
// RUN: %{run} %t.out

#define KERNEL_CLASS_DECL class KernelFunctor;
#define LIB_A "lib_a.dll"
#define LIB_B "lib_b.dll"
#include "../kernel_name_collision_kernel_bundle.inc"
