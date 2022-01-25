// RUN: %clang_cc1 %s -triple spir -cl-std=CL1.2 -emit-llvm-bc -fdeclare-opencl-builtins -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.spv -r --spirv-target-env=CL1.2 -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

// This test checks that the translator is capable to correctly translate
// atomic_cmpxchg OpenCL C 1.2 built-in function [1] into corresponding SPIR-V
// instruction and vice-versa.

__kernel void test_atomic_cmpxchg(__global int *p, int cmp, int val) {
  atomic_cmpxchg(p, cmp, val);

  __global unsigned int *up = (__global unsigned int *)p;
  unsigned int ucmp = (unsigned int)cmp;
  unsigned int uval = (unsigned int)val;
  atomic_cmpxchg(up, ucmp, uval);
}

// CHECK-SPIRV: Name [[TEST:[0-9]+]] "test_atomic_cmpxchg"
// CHECK-SPIRV-DAG: TypeInt [[UINT:[0-9]+]] 32 0
// CHECK-SPIRV-DAG: TypePointer [[UINT_PTR:[0-9]+]] 5 [[UINT]]
//
// In SPIR-V, atomic_cmpxchg is represented as OpAtomicCompareExchange [2],
// which also includes memory scope and two memory semantic arguments. The
// translator applies some default memory order for it and therefore, constants
// below include a bit more information than original source
//
// 0x2 Workgroup
// CHECK-SPIRV-DAG: Constant [[UINT]] [[WORKGROUP_SCOPE:[0-9]+]] 2
//
// 0x0 Relaxed
// TODO: do we need CrossWorkgroupMemory here as well?
// CHECK-SPIRV-DAG: Constant [[UINT]] [[RELAXED:[0-9]+]] 0
//
// CHECK-SPIRV: Function {{[0-9]+}} [[TEST]]
// CHECK-SPIRV: FunctionParameter [[UINT_PTR]] [[PTR:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[UINT]] [[CMP:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[UINT]] [[VAL:[0-9]+]]
// CHECK-SPIRV: AtomicCompareExchange [[UINT]] {{[0-9]+}} [[PTR]] [[WORKGROUP_SCOPE]] [[RELAXED]] [[RELAXED]] [[VAL]] [[CMP]]
// CHECK-SPIRV: AtomicCompareExchange [[UINT]] {{[0-9]+}} [[PTR]] [[WORKGROUP_SCOPE]] [[RELAXED]] [[RELAXED]] [[VAL]] [[CMP]]
//
//
// CHECK-LLVM-LABEL: define spir_kernel void @test_atomic_cmpxchg
// CHECK-LLVM: call spir_func i32 @_Z14atomic_cmpxchgPU3AS1Viii
// TODO: is it an issue that we lost call to @_Z14atomic_cmpxchgPU3AS1jjj here?
// CHECK-LLVM: call spir_func i32 @_Z14atomic_cmpxchgPU3AS1Viii

// References:
// [1]: https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/atomic_cmpxchg.html
// [2]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpAtomicCompareExchange
