// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-spirv -opaque-pointers -r -emit-opaque-pointers %t.spv -o %t.rev.bc
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-dis -opaque-pointers < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

kernel void testBitOperations(int a, int b, int c, global int *res) {
  *res = (a & b) | (0x6F ^ c);
  *res += popcount(b);
}

// CHECK-SPIRV: BitwiseAnd
// CHECK-SPIRV: BitwiseXor
// CHECK-SPIRV: BitwiseOr
// CHECK-SPIRV: BitCount

// CHECK-LLVM-LABEL: @testBitOperations
// CHECK-LLVM: and i32
// CHECK-LLVM: xor i32
// CHECK-LLVM: or i32
// CHECK-LLVM: call spir_func i32 @_Z8popcounti(i32 %b)
