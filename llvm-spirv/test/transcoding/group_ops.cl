// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

// CHECK-SPIRV-DAG: TypeInt [[int:[0-9]+]] 32 0
// CHECK-SPIRV-DAG: TypeFloat [[float:[0-9]+]] 32
// CHECK-SPIRV-DAG: Constant [[int]] [[ScopeWorkgroup:[0-9]+]] 2
// CHECK-SPIRV-DAG: Constant [[int]] [[ScopeSubgroup:[0-9]+]] 3

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupFMax [[float]] {{[0-9]+}} [[ScopeWorkgroup]] 0
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupFMax
// CHECK-LLVM: call spir_func float @_Z21work_group_reduce_maxf(float %a)

kernel void testWorkGroupFMax(float a, global float *res) {
  res[0] = work_group_reduce_max(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupFMin [[float]] {{[0-9]+}} [[ScopeWorkgroup]] 0
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupFMin
// CHECK-LLVM: call spir_func float @_Z21work_group_reduce_minf(float %a)

kernel void testWorkGroupFMin(float a, global float *res) {
  res[0] = work_group_reduce_min(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupFAdd [[float]] {{[0-9]+}} [[ScopeWorkgroup]] 0
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupFAdd
// CHECK-LLVM: call spir_func float @_Z21work_group_reduce_addf(float %a)

kernel void testWorkGroupFAdd(float a, global float *res) {
  res[0] = work_group_reduce_add(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupFMax [[float]] {{[0-9]+}} [[ScopeWorkgroup]] 1
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupScanInclusiveFMax
// CHECK-LLVM: call spir_func float @_Z29work_group_scan_inclusive_maxf(float %a)

kernel void testWorkGroupScanInclusiveFMax(float a, global float *res) {
  res[0] = work_group_scan_inclusive_max(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupFMax [[float]] {{[0-9]+}} [[ScopeWorkgroup]] 2
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupScanExclusiveFMax
// CHECK-LLVM: call spir_func float @_Z29work_group_scan_exclusive_maxf(float %a)

kernel void testWorkGroupScanExclusiveFMax(float a, global float *res) {
  res[0] = work_group_scan_exclusive_max(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupSMax [[int]] {{[0-9]+}} [[ScopeWorkgroup]] 0
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupSMax
// CHECK-LLVM: call spir_func i32 @_Z21work_group_reduce_maxi(i32 %a)

kernel void testWorkGroupSMax(int a, global int *res) {
  res[0] = work_group_reduce_max(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupSMin [[int]] {{[0-9]+}} [[ScopeWorkgroup]] 0
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupSMin
// CHECK-LLVM: call spir_func i32 @_Z21work_group_reduce_mini(i32 %a)

kernel void testWorkGroupSMin(int a, global int *res) {
  res[0] = work_group_reduce_min(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupIAdd [[int]] {{[0-9]+}} [[ScopeWorkgroup]] 0
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupIAddSigned
// TODO: This should map to _Z21work_group_reduce_addj, instead.
// Update this test and remove OpGroupIAdd.spt when fixing this.
// CHECK-LLVM: call spir_func i32 @_Z21work_group_reduce_addi(i32 %a)

kernel void testWorkGroupIAddSigned(int a, global int *res) {
  res[0] = work_group_reduce_add(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupIAdd [[int]] {{[0-9]+}} [[ScopeWorkgroup]] 0
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupIAddUnsigned
// TODO: This should map to _Z21work_group_reduce_addj, instead.
// Update this test and remove OpGroupIAdd.spt when fixing this.
// CHECK-LLVM: call spir_func i32 @_Z21work_group_reduce_addi(i32 %a)

kernel void testWorkGroupIAddUnsigned(uint a, global uint *res) {
  res[0] = work_group_reduce_add(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupUMax [[int]] {{[0-9]+}} [[ScopeWorkgroup]] 0
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupUMax
// CHECK-LLVM: call spir_func i32 @_Z21work_group_reduce_maxj(i32 %a)

kernel void testWorkGroupUMax(uint a, global uint *res) {
  res[0] = work_group_reduce_max(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupUMax [[int]] {{[0-9]+}} [[ScopeSubgroup]] 0
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testSubGroupUMax
// CHECK-LLVM: call spir_func i32 @_Z20sub_group_reduce_maxj(i32 %a)

#pragma OPENCL EXTENSION cl_khr_subgroups: enable
kernel void testSubGroupUMax(uint a, global uint *res) {
  res[0] = sub_group_reduce_max(a);
}
#pragma OPENCL EXTENSION cl_khr_subgroups: disable

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupUMax [[int]] {{[0-9]+}} [[ScopeWorkgroup]] 1
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupScanInclusiveUMax
// CHECK-LLVM: call spir_func i32 @_Z29work_group_scan_inclusive_maxj(i32 %a)

kernel void testWorkGroupScanInclusiveUMax(uint a, global uint *res) {
  res[0] = work_group_scan_inclusive_max(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupUMax [[int]] {{[0-9]+}} [[ScopeWorkgroup]] 2
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupScanExclusiveUMax
// CHECK-LLVM: call spir_func i32 @_Z29work_group_scan_exclusive_maxj(i32 %a)

kernel void testWorkGroupScanExclusiveUMax(uint a, global uint *res) {
  res[0] = work_group_scan_exclusive_max(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupUMin [[int]] {{[0-9]+}} [[ScopeWorkgroup]] 0
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupUMin
// CHECK-LLVM: call spir_func i32 @_Z21work_group_reduce_minj(i32 %a)

kernel void testWorkGroupUMin(uint a, global uint *res) {
  res[0] = work_group_reduce_min(a);
}

// CHECK-SPIRV-LABEL: 5 Function
// CHECK-SPIRV: GroupBroadcast [[int]] {{[0-9]+}} [[ScopeWorkgroup]]
// CHECK-SPIRV: FunctionEnd

// CHECK-LLVM-LABEL: @testWorkGroupBroadcast
// CHECK-LLVM: call spir_func i32 @_Z20work_group_broadcast{{[ji]}}{{[jm]}}(i32 %a, i32 %0)

kernel void testWorkGroupBroadcast(uint a, global size_t *id, global int *res) {
  res[0] = work_group_broadcast(a, *id);
}
