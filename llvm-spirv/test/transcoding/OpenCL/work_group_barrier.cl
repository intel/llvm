// RUN: %clang_cc1 %s -triple spir -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.spv -r --spirv-target-env=CL2.0 -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

// This test checks that the translator is capable to correctly translate
// sub_group_barrier built-in function [1] from cl_khr_subgroups extension into
// corresponding SPIR-V instruction and vice-versa.

__kernel void test_barrier_const_flags() {
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  work_group_barrier(CLK_IMAGE_MEM_FENCE);

  work_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  work_group_barrier(CLK_LOCAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE);
  work_group_barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE);

  work_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_work_item);
  work_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_work_group);
  work_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_device);
  work_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_all_svm_devices);
  work_group_barrier(CLK_LOCAL_MEM_FENCE, memory_scope_sub_group);

  // barrier should also work (preserved for backward compatibility)
  barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void test_barrier_non_const_flags(cl_mem_fence_flags flags, memory_scope scope) {
  // FIXME: OpenCL spec doesn't require flags to be compile-time known
  // work_group_barrier(flags);
  // work_group_barrier(flags, scope);
}

// CHECK-SPIRV: EntryPoint {{[0-9]+}} [[TEST_CONST_FLAGS:[0-9]+]] "test_barrier_const_flags"
// CHECK-SPIRV: TypeInt [[UINT:[0-9]+]] 32 0
//
// In SPIR-V, barrier is represented as OpControlBarrier [2] and OpenCL
// cl_mem_fence_flags are represented as part of Memory Semantics [3], which
// also includes memory order constraints. The translator applies some default
// memory order for OpControlBarrier and therefore, constants below include a
// bit more information than original source
//
// 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[LOCAL:[0-9]+]] 272
// 0x10 SequentiallyConsistent + 0x200 CrossWorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[GLOBAL:[0-9]+]] 528
// 0x10 SequentiallyConsistent + 0x800 ImageMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[IMAGE:[0-9]+]] 2064
// 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[LOCAL_GLOBAL:[0-9]+]] 784
// 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory + 0x800 ImageMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[LOCAL_IMAGE:[0-9]+]] 2320
// 0x10 SequentiallyConsistent + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory + 0x800 ImageMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[LOCAL_GLOBAL_IMAGE:[0-9]+]] 2832
//
// Scopes [4]:
// 2 Workgroup
// CHECK-SPIRV-DAG: Constant [[UINT]] [[SCOPE_WORK_GROUP:[0-9]+]] 2
// 4 Invocation
// CHECK-SPIRV-DAG: Constant [[UINT]] [[SCOPE_INVOCATION:[0-9]+]] 4
// 1 Device
// CHECK-SPIRV-DAG: Constant [[UINT]] [[SCOPE_DEVICE:[0-9]+]] 1
// 0 CrossDevice
// CHECK-SPIRV-DAG: Constant [[UINT]] [[SCOPE_CROSS_DEVICE:[0-9]+]] 0
// 3 Subgroup
// CHECK-SPIRV-DAG: Constant [[UINT]] [[SCOPE_SUBGROUP:[0-9]+]] 3
//
// CHECK-SPIRV: Function {{[0-9]+}} [[TEST_CONST_FLAGS]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[LOCAL]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[GLOBAL]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[IMAGE]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[LOCAL_GLOBAL]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[LOCAL_IMAGE]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[LOCAL_GLOBAL_IMAGE]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_INVOCATION]] [[LOCAL]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[LOCAL]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_DEVICE]] [[LOCAL]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_CROSS_DEVICE]] [[LOCAL]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_SUBGROUP]] [[LOCAL]]
// CHECK-SPIRV: ControlBarrier [[SCOPE_WORK_GROUP]] [[SCOPE_WORK_GROUP]] [[GLOBAL]]
//
// CHECK-LLVM-LABEL: define spir_kernel void @test_barrier_const_flags
// CHECK-LLVM: call spir_func void @_Z18work_group_barrierj12memory_scope(i32 1, i32 1)
// CHECK-LLVM: call spir_func void @_Z18work_group_barrierj12memory_scope(i32 2, i32 1)
// CHECK-LLVM: call spir_func void @_Z18work_group_barrierj12memory_scope(i32 4, i32 1)
// CHECK-LLVM: call spir_func void @_Z18work_group_barrierj12memory_scope(i32 3, i32 1)
// CHECK-LLVM: call spir_func void @_Z18work_group_barrierj12memory_scope(i32 5, i32 1)
// CHECK-LLVM: call spir_func void @_Z18work_group_barrierj12memory_scope(i32 7, i32 1)
// CHECK-LLVM: call spir_func void @_Z18work_group_barrierj12memory_scope(i32 1, i32 1)
// CHECK-LLVM: call spir_func void @_Z18work_group_barrierj12memory_scope(i32 1, i32 2)
// CHECK-LLVM: call spir_func void @_Z18work_group_barrierj12memory_scope(i32 1, i32 3)
// CHECK-LLVM: call spir_func void @_Z18work_group_barrierj12memory_scope(i32 1, i32 4)
// CHECK-LLVM: call spir_func void @_Z18work_group_barrierj12memory_scope(i32 2, i32 1)

// References:
// [1]: https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/work_group_barrier.html
// [2]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpControlBarrier
// [3]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_memory_semantics__id_a_memory_semantics_lt_id_gt
