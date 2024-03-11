// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-SPIRV-DAG: TypeInt [[int:[0-9]+]] 32 0
// CHECK-SPIRV-DAG: TypeVector [[int2:[0-9]+]] [[int]] 2
// CHECK-SPIRV-DAG: TypeFloat [[float:[0-9]+]] 32
// CHECK-SPIRV-DAG: TypeVector [[float2:[0-9]+]] [[float]] 2

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
// CHECK-SPIRV: SDiv [[int2]] {{[0-9]+}} [[A]] [[B]]
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testSDiv
// CHECK-LLVM: sdiv <2 x i32> %a, %b

kernel void testSDiv(int2 a, int2 b, global int2 *res) {
  res[0] = a / b;
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
// CHECK-SPIRV: UDiv [[int2]] {{[0-9]+}} [[A]] [[B]]
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testUDiv
// CHECK-LLVM: udiv <2 x i32> %a, %b

kernel void testUDiv(uint2 a, uint2 b, global uint2 *res) {
  res[0] = a / b;
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
// CHECK-SPIRV: FDiv [[float2]] {{[0-9]+}} [[A]] [[B]]
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testFDiv
// CHECK-LLVM: fdiv <2 x float> %a, %b

kernel void testFDiv(float2 a, float2 b, global float2 *res) {
  res[0] = a / b;
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
// CHECK-SPIRV: SRem [[int2]] {{[0-9]+}} [[A]] [[B]]
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testSRem
// CHECK-LLVM: srem <2 x i32> %a, %b

kernel void testSRem(int2 a, int2 b, global int2 *res) {
  res[0] = a % b;
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
// CHECK-SPIRV-NEXT: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
// CHECK-SPIRV: UMod [[int2]] {{[0-9]+}} [[A]] [[B]]
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testUMod
// CHECK-LLVM: urem <2 x i32> %a, %b

kernel void testUMod(uint2 a, uint2 b, global uint2 *res) {
  res[0] = a % b;
}
