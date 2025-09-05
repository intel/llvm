// RUN: %clang_cc1 -triple spir-unknown-unknown -O0 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-SPIRV: FNegate
// CHECK-SPIRV: FNegate
// CHECK-SPIRV: FNegate
// CHECK-SPIRV: FNegate

// CHECK-LLVM: fneg half %
// CHECK-LLVM: fneg float %
// CHECK-LLVM: fneg double %
// CHECK-LLVM: fneg <8 x double> %

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void foo(double a1, __global half *h, __global float *b0, __global double *b1, __global double8 *d) {
   *h = -*h;
   *b0 = -*b0;
   *b1 = -a1;
   *d = -*d;
}
