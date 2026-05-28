// REQUIRES: windows
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %{build} -DBUILD_LIB_ADD %shared_lib -o %t.dir/lib_add.dll
// RUN: %{build} -DBUILD_LIB_MUL %shared_lib -o %t.dir/lib_mul.dll
// RUN: %{build} -DBUILD_MAIN '-DLIB_DIR="%/t.dir"' -o %t.out
// RUN: %{run} %t.out

#define KERNEL_CLASS_DECL class KernelFunctor;
#define LIB_ADD "lib_add.dll"
#define LIB_MUL "lib_mul.dll"
#include "../kernel_name_collision_same_layout.inc"
