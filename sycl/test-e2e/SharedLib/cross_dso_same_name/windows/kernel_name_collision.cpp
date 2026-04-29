// REQUIRES: windows
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %{build} -DBUILD_SMALL_LIB %shared_lib -o %t.dir/small.dll
// RUN: %{build} -DBUILD_LARGE_LIB %shared_lib -o %t.dir/large.dll
// RUN: %{build} -DBUILD_MAIN '-DLIB_DIR="%/t.dir"' -o %t.out
// RUN: %{run} %t.out

#define KERNEL_CLASS_DECL class KernelFunctor;
#define LIB_SMALL "small.dll"
#define LIB_LARGE "large.dll"
#include "../kernel_name_collision.inc"
