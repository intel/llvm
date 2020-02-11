// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-SPIRV: SatConvertSToU

// CHECK-LLVM-LABEL: @testSToU
// CHECK-LLVM: call spir_func <2 x i8> @_Z18convert_uchar2_satDv2_i

kernel void testSToU(global int2 *a, global uchar2 *res) {
  res[0] = convert_uchar2_sat(*a);
}

// CHECK-SPIRV: SatConvertUToS

// CHECK-LLVM-LABEL: @testUToS
// CHECK-LLVM: call spir_func <2 x i8> @_Z17convert_char2_satDv2_j

kernel void testUToS(global uint2 *a, global char2 *res) {
  res[0] = convert_char2_sat(*a);
}
