// RUN: %clang_cc1 %s -triple spir -cl-std=CL1.2 -emit-llvm-bc -fdeclare-opencl-builtins -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.spv -r --spirv-target-env=CL1.2 -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

// This test checks that the translator is capable to correctly translate
// legacy atomic OpenCL C 1.2 built-in functions [1] into corresponding SPIR-V
// instruction and vice-versa.

__kernel void test_legacy_atomics(__global int *p, int val) {
  atom_add(p, val);     // from cl_khr_global_int32_base_atomics
  atomic_add(p, val);   // from OpenCL C 1.1
}

// CHECK-SPIRV: Name [[TEST:[0-9]+]] "test_legacy_atomics"
// CHECK-SPIRV-DAG: TypeInt [[UINT:[0-9]+]] 32 0
// CHECK-SPIRV-DAG: TypePointer [[UINT_PTR:[0-9]+]] 5 [[UINT]]
//
// In SPIR-V, atomic_add is represented as OpAtomicIAdd [2], which also includes
// memory scope and memory semantic arguments. The translator applies a default
// memory scope and memory order for it and therefore, constants below include
// a bit more information than original source
//
// 0x2 Workgroup
// CHECK-SPIRV-DAG: Constant [[UINT]] [[WORKGROUP_SCOPE:[0-9]+]] 2
//
// 0x0 Relaxed
// CHECK-SPIRV-DAG: Constant [[UINT]] [[RELAXED:[0-9]+]] 0
//
// CHECK-SPIRV: Function {{[0-9]+}} [[TEST]]
// CHECK-SPIRV: FunctionParameter [[UINT_PTR]] [[PTR:[0-9]+]]
// CHECK-SPIRV: FunctionParameter [[UINT]] [[VAL:[0-9]+]]
// CHECK-SPIRV: AtomicIAdd [[UINT]] {{[0-9]+}} [[PTR]] [[WORKGROUP_SCOPE]] [[RELAXED]] [[VAL]]
// CHECK-SPIRV: AtomicIAdd [[UINT]] {{[0-9]+}} [[PTR]] [[WORKGROUP_SCOPE]] [[RELAXED]] [[VAL]]
//
//
// CHECK-LLVM-LABEL: define spir_kernel void @test_legacy_atomics
// Note: the translator generates the OpenCL C 1.1 function name exclusively!
// CHECK-LLVM: call spir_func i32 @_Z10atomic_addPU3AS1Vii
// CHECK-LLVM: call spir_func i32 @_Z10atomic_addPU3AS1Vii

// References:
// [1]: https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_C.html#atomic-legacy
// [2]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpAtomicIAdd
