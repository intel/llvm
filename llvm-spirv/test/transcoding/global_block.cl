// There are no blocks in SPIR-V. Therefore they are translated into regular
// functions. An LLVM module which uses blocks, also contains some auxiliary
// block-specific instructions, which are redundant in SPIR-V and should be
// removed

// RUN: %clang_cc1 -O0 -triple spir-unknown-unknown -cl-std=CL2.0 -x cl %s -emit-llvm-bc -o %t.bc
// TODO: currently max version is limited to 1.1 for this test. Issues here
// that the SPIR-V module generated for blocks is invalid for versions starting
// from 1.4, spirv-val is failing with:
//   error: line 63: Interface variable id <13> is used by entry point
//                   'block_kernel' id <24>, but is not listed as an interface
//   %__block_literal_global = OpVariable %_ptr_CrossWorkgroup__struct_10
//                         CrossWorkgroup %11
// details can be found in:
// – Public issue #35: OpEntryPoint must list all global variables in the
//   interface. Additionally, duplication in the list is not allowed.
// RUN: llvm-spirv --spirv-max-version=1.1 %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv --spirv-max-version=1.1 %t.bc -o %t.spv
// RUN: spirv-val %t.spv
// RUN: llvm-spirv -r %t.spv -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-LLVM

kernel void block_kernel(__global int* res) {
  typedef int (^block_t)(int);
  constant block_t b1 = ^(int i) { return i + 1; };
  *res = b1(5);
}

// CHECK-SPIRV: Name [[block_invoke:[0-9]+]] "_block_invoke"
// CHECK-SPIRV: TypeInt [[int:[0-9]+]] 32
// CHECK-SPIRV: TypeInt [[int8:[0-9]+]] 8
// CHECK-SPIRV: Constant [[int]] [[five:[0-9]+]] 5
// CHECK-SPIRV: TypePointer [[int8Ptr:[0-9]+]] 8 [[int8]]
// CHECK-SPIRV: TypeFunction [[block_invoke_type:[0-9]+]] [[int]] [[int8Ptr]] [[int]]

// CHECK-LLVM-LABEL: @block_kernel
// CHECK-SPIRV: FunctionCall [[int]] {{[0-9]+}} [[block_invoke]] {{[0-9]+}} [[five]]
// CHECK-LLVM: %call = call spir_func i32 @_block_invoke(i8 addrspace(4)* {{.*}}, i32 5)

// CHECK-SPIRV: 5 Function [[int]] [[block_invoke]] 2 [[block_invoke_type]]
// CHECK-SPIRV-NEXT: 3 FunctionParameter [[int8Ptr]] {{[0-9]+}}
// CHECK-SPIRV-NEXT: 3 FunctionParameter [[int]] {{[0-9]+}}
// CHECK-LLVM: define internal spir_func i32 @_block_invoke(i8 addrspace(4)* {{.*}}, i32 %{{.*}})
