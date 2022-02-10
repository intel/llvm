// RUN: %clang_cc1 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -fcuda-prec-sqrt %s -o -| FileCheck --check-prefix=CHECK-ON %s
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm %s -o -| FileCheck --check-prefix=CHECK-OFF %s

#include "Inputs/cuda.h"

// Check that the -fcuda-prec-sqrt flag correctly sets the nvvm-reflect module flags.

extern "C" __device__ void foo() {}

// CHECK-ON: !{i32 4, !"nvvm-reflect-prec-sqrt", i32 1}
// CHECK-OFF: !{i32 4, !"nvvm-reflect-prec-sqrt", i32 0}
