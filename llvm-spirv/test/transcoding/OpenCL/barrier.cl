// RUN: %clang_cc1 %s -triple spir -cl-std=CL1.2 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.spv -r --spirv-target-env=CL1.2 -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

// This test checks that the translator is capable to correctly translate
// barrier OpenCL C 1.2 built-in function [1] into corresponding SPIR-V
// instruction and vice-versa.

// FIXME: Strictly speaking, this flag is not supported by barrier in OpenCL 1.2
#define CLK_IMAGE_MEM_FENCE 0x04

void __attribute__((overloadable)) __attribute__((convergent)) barrier(cl_mem_fence_flags);

__kernel void test_barrier_const_flags() {
  barrier(CLK_LOCAL_MEM_FENCE);
  barrier(CLK_GLOBAL_MEM_FENCE);
  barrier(CLK_IMAGE_MEM_FENCE);

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  barrier(CLK_LOCAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE);
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE);
}

__kernel void test_barrier_non_const_flags(cl_mem_fence_flags flags) {
  // FIXME: OpenCL spec doesn't require flags to be compile-time known
  // barrier(flags);
}

// CHECK-SPIRV: Name [[TEST_CONST_FLAGS:[0-9]+]] "test_barrier_const_flags"
// CHECK-SPIRV: TypeInt [[UINT:[0-9]+]] 32 0
//
// In SPIR-V, barrier is represented as OpControlBarrier [3] and OpenCL
// cl_mem_fence_flags are represented as part of Memory Semantics [2], which
// also includes memory order constraints. The translator applies some default
// memory order for OpControlBarrier and therefore, constants below include a
// bit more information than original source
//
// 0x2 Workgroup
// CHECK-SPIRV-DAG: Constant [[UINT]] [[WG:[0-9]+]] 2
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
// CHECK-SPIRV: Function {{[0-9]+}} [[TEST_CONST_FLAGS]]
// CHECK-SPIRV: ControlBarrier [[WG]] [[WG]] [[LOCAL]]
// CHECK-SPIRV: ControlBarrier [[WG]] [[WG]] [[GLOBAL]]
// CHECK-SPIRV: ControlBarrier [[WG]] [[WG]] [[IMAGE]]
// CHECK-SPIRV: ControlBarrier [[WG]] [[WG]] [[LOCAL_GLOBAL]]
// CHECK-SPIRV: ControlBarrier [[WG]] [[WG]] [[LOCAL_IMAGE]]
// CHECK-SPIRV: ControlBarrier [[WG]] [[WG]] [[LOCAL_GLOBAL_IMAGE]]
//
// CHECK-LLVM-LABEL: define spir_kernel void @test_barrier_const_flags
// CHECK-LLVM: call spir_func void @_Z7barrierj(i32 1)
// CHECK-LLVM: call spir_func void @_Z7barrierj(i32 2)
// CHECK-LLVM: call spir_func void @_Z7barrierj(i32 4)
// CHECK-LLVM: call spir_func void @_Z7barrierj(i32 3)
// CHECK-LLVM: call spir_func void @_Z7barrierj(i32 5)
// CHECK-LLVM: call spir_func void @_Z7barrierj(i32 7)

// References:
// [1]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/barrier.html
// [2]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_memory_semantics__id_a_memory_semantics_lt_id_gt
// [3]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpControlBarrier
