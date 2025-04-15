// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: not llvm-spirv --spirv-max-version=1.4 %t.bc 2>&1 | FileCheck --check-prefix=CHECK-ERROR %s
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

// Before SPIR-V 1.5, the Id operand of OpGroupNonUniformBroadcast must come from a constant instruction.
// CHECK-ERROR: RequiresVersion: Cannot fulfill SPIR-V version restriction:
// CHECK-ERROR-NEXT: SPIR-V version was restricted to at most 1.4 (66560) but a construct from the input requires SPIR-V version 1.5 (66816) or above

// CHECK-LLVM-LABEL: @test
// CHECK-LLVM: call spir_func i16 @_Z31sub_group_non_uniform_broadcasttj(i16 %a, i32 %id)

kernel void test(short a, uint id, global short *res) {
  res[0] = sub_group_non_uniform_broadcast(a, id);
}
