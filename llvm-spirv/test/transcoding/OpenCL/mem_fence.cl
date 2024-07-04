// RUN: %clang_cc1 %s -triple spir -cl-std=CL1.2 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.spv -r --spirv-target-env=CL1.2 -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

// This test checks that the translator is capable to correctly translate
// mem_fence OpenCL C 1.2 built-in function [1] into corresponding SPIR-V
// instruction and vice-versa.

// Strictly speaking, this flag is not supported by mem_fence in OpenCL 1.2
#define CLK_IMAGE_MEM_FENCE 0x04

__kernel void test_mem_fence_const_flags() {
  mem_fence(CLK_LOCAL_MEM_FENCE);
  mem_fence(CLK_GLOBAL_MEM_FENCE);
  mem_fence(CLK_IMAGE_MEM_FENCE);

  mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  mem_fence(CLK_LOCAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE);
  mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE);

  read_mem_fence(CLK_LOCAL_MEM_FENCE);
  read_mem_fence(CLK_GLOBAL_MEM_FENCE);
  read_mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  write_mem_fence(CLK_LOCAL_MEM_FENCE);
  write_mem_fence(CLK_GLOBAL_MEM_FENCE);
  write_mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

__kernel void test_mem_fence_non_const_flags(cl_mem_fence_flags flags) {
  // FIXME: OpenCL spec doesn't require flags to be compile-time known
  // mem_fence(flags);
}

// CHECK-SPIRV: Name [[TEST_CONST_FLAGS:[0-9]+]] "test_mem_fence_const_flags"
// CHECK-SPIRV: TypeInt [[UINT:[0-9]+]] 32 0
//
// In SPIR-V, mem_fence is represented as OpMemoryBarrier [2] and OpenCL
// cl_mem_fence_flags are represented as part of Memory Semantics [3], which
// also includes memory order constraints. The translator applies some default
// memory order for OpMemoryBarrier and therefore, constants below include a
// bit more information than original source
//
// 0x2 Workgroup
// CHECK-SPIRV-DAG: Constant [[UINT]] [[WG:[0-9]+]] 2
//
// 0x2 Acquire + 0x100 WorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQ_LOCAL:[0-9]+]] 258
// 0x2 Acquire + 0x200 CrossWorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQ_GLOBAL:[0-9]+]] 514
// 0x2 Acquire + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQ_LOCAL_GLOBAL:[0-9]+]] 770
//
// 0x4 Release + 0x100 WorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[REL_LOCAL:[0-9]+]] 260
// 0x4 Release + 0x200 CrossWorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[REL_GLOBAL:[0-9]+]] 516
// 0x4 Release + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[REL_LOCAL_GLOBAL:[0-9]+]] 772
//
// 0x8 AcquireRelease + 0x100 WorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQ_REL_LOCAL:[0-9]+]] 264
// 0x8 AcquireRelease + 0x200 CrossWorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQ_REL_GLOBAL:[0-9]+]] 520
// 0x8 AcquireRelease + 0x800 ImageMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQ_REL_IMAGE:[0-9]+]] 2056
// 0x8 AcquireRelease + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQ_REL_LOCAL_GLOBAL:[0-9]+]] 776
// 0x8 AcquireRelease + 0x100 WorkgroupMemory + 0x800 ImageMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQ_REL_LOCAL_IMAGE:[0-9]+]] 2312
// 0x8 AcquireRelease + 0x100 WorkgroupMemory + 0x200 CrossWorkgroupMemory + 0x800 ImageMemory
// CHECK-SPIRV-DAG: Constant [[UINT]] [[ACQ_REL_LOCAL_GLOBAL_IMAGE:[0-9]+]] 2824
//
// CHECK-SPIRV: Function {{[0-9]+}} [[TEST_CONST_FLAGS]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[ACQ_REL_LOCAL]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[ACQ_REL_GLOBAL]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[ACQ_REL_IMAGE]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[ACQ_REL_LOCAL_GLOBAL]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[ACQ_REL_LOCAL_IMAGE]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[ACQ_REL_LOCAL_GLOBAL_IMAGE]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[ACQ_LOCAL]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[ACQ_GLOBAL]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[ACQ_LOCAL_GLOBAL]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[REL_LOCAL]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[REL_GLOBAL]]
// CHECK-SPIRV: MemoryBarrier [[WG]] [[REL_LOCAL_GLOBAL]]
//
// CHECK-LLVM-LABEL: define spir_kernel void @test_mem_fence_const_flags
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 1)
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 2)
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 4)
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 3)
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 5)
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 7)
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 1)
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 2)
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 3)
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 1)
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 2)
// CHECK-LLVM: call spir_func void @_Z9mem_fencej(i32 3)

// References:
// [1]: https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/mem_fence.html
// [2]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpMemoryBarrier
// [3]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_memory_semantics__id_a_memory_semantics_lt_id_gt
