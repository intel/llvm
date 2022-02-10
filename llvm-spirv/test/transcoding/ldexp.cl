// Check that translator converts scalar arg to vector for ldexp math instructions
// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -to-text %t.spv -o %t.spt
// RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM-CL20

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void test_kernel_half(half3 x, int k, __global half3* ret) {
   *ret = ldexp(x, k);
}

// CHECK-SPIRV: {{.*}} ldexp
// CHECK-LLVM-CL20: %call = call spir_func <3 x half> @_Z5ldexpDv3_DhDv3_i(

__kernel void test_kernel_float(float3 x, int k, __global float3* ret) {
   *ret = ldexp(x, k);
}

// CHECK-SPIRV: {{.*}} ldexp
// CHECK-LLVM-CL20: %call = call spir_func <3 x float> @_Z5ldexpDv3_fDv3_i(

__kernel void test_kernel_double(double3 x, int k, __global double3* ret) {
   *ret = ldexp(x, k);
}

// CHECK-SPIRV: {{.*}} ldexp
// CHECK-LLVM-CL20: %call = call spir_func <3 x double> @_Z5ldexpDv3_dDv3_i(
