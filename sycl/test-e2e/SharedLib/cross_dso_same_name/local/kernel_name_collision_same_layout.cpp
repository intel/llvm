// REQUIRES: linux
//
// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: %{build} -DBUILD_LIB_ADD -fPIC -shared -o %t.dir/lib_add.so
// RUN: %{build} -DBUILD_LIB_MUL -fPIC -shared -o %t.dir/lib_mul.so
// RUN: %{build} -DBUILD_MAIN '-DLIB_DIR="%t.dir"' -ldl -o %t.out
// RUN: %{run} %t.out

#define DLOPEN_FLAGS (RTLD_NOW | RTLD_LOCAL)
#define KERNEL_CLASS_DECL class KernelFunctor;
#define LIB_ADD "lib_add.so"
#define LIB_MUL "lib_mul.so"
#include "../kernel_name_collision_same_layout.inc"
