// RUN: %clang_cc1 -triple spir -cl-std=cl2.0 %s -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc -o %t.bc
// RUN: llvm-spirv %t.bc -o %t.spv
// RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.spv -r --spirv-target-env=CL2.0 -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

int load (volatile atomic_int* obj, memory_order order, memory_scope scope) {
  return atomic_load_explicit(obj, order, scope);
}

// CHECK-SPIRV: Name [[LOAD:[0-9]+]] "load"
// CHECK-SPIRV: Name [[TRANS_MEM_SCOPE:[0-9]+]] "__translate_ocl_memory_scope"
// CHECK-SPIRV: Name [[TRANS_MEM_ORDER:[0-9]+]] "__translate_ocl_memory_order"

// CHECK-SPIRV: TypeInt [[int:[0-9]+]] 32 0
// CHECK-SPIRV-DAG: Constant [[int]] [[ZERO:[0-9]+]] 0
// CHECK-SPIRV-DAG: Constant [[int]] [[ONE:[0-9]+]] 1
// CHECK-SPIRV-DAG: Constant [[int]] [[TWO:[0-9]+]] 2
// CHECK-SPIRV-DAG: Constant [[int]] [[THREE:[0-9]+]] 3
// CHECK-SPIRV-DAG: Constant [[int]] [[FOUR:[0-9]+]] 4
// CHECK-SPIRV-DAG: Constant [[int]] [[EIGHT:[0-9]+]] 8
// CHECK-SPIRV-DAG: Constant [[int]] [[SIXTEEN:[0-9]+]] 16

// CHECK-SPIRV: Function {{[0-9]+}} [[LOAD]]
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[OBJECT:[0-9]+]]
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[OCL_ORDER:[0-9]+]]
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[OCL_SCOPE:[0-9]+]]

// CHECK-SPIRV: FunctionCall [[int]] [[SPIRV_SCOPE:[0-9]+]] [[TRANS_MEM_SCOPE]] [[OCL_SCOPE]]
// CHECK-SPIRV: FunctionCall [[int]] [[SPIRV_ORDER:[0-9]+]] [[TRANS_MEM_ORDER]] [[OCL_ORDER]]
// CHECK-SPIRV: AtomicLoad [[int]] {{[0-9]+}} [[OBJECT]] [[SPIRV_SCOPE]] [[SPIRV_ORDER]]

// CHECK-SPIRV: Function [[int]] [[TRANS_MEM_SCOPE]]
// CHECK-SPIRV: FunctionParameter [[int]] [[KEY:[0-9]+]]
// CHECK-SPIRV: Switch [[KEY]] [[CASE_2:[0-9]+]] 0 [[CASE_0:[0-9]+]] 1 [[CASE_1:[0-9]+]] 2 [[CASE_2]] 3 [[CASE_3:[0-9]+]] 4 [[CASE_4:[0-9]+]]
// CHECK-SPIRV: Label [[CASE_0]]
// CHECK-SPIRV: ReturnValue [[FOUR]]
// CHECK-SPIRV: Label [[CASE_1]]
// CHECK-SPIRV: ReturnValue [[TWO]]
// CHECK-SPIRV: Label [[CASE_2]]
// CHECK-SPIRV: ReturnValue [[ONE]]
// CHECK-SPIRV: Label [[CASE_3]]
// CHECK-SPIRV: ReturnValue [[ZERO]]
// CHECK-SPIRV: Label [[CASE_4]]
// CHECK-SPIRV: ReturnValue [[THREE]]
// CHECK-SPIRV: FunctionEnd

// CHECK-SPIRV: Function [[int]] [[TRANS_MEM_ORDER]]
// CHECK-SPIRV: FunctionParameter [[int]] [[KEY:[0-9]+]]
// CHECK-SPIRV: Switch [[KEY]] [[CASE_5:[0-9]+]] 0 [[CASE_0:[0-9]+]] 2 [[CASE_2:[0-9]+]] 3 [[CASE_3:[0-9]+]] 4 [[CASE_4:[0-9]+]] 5 [[CASE_5]]
// CHECK-SPIRV: Label [[CASE_0]]
// CHECK-SPIRV: ReturnValue [[ZERO]]
// CHECK-SPIRV: Label [[CASE_2]]
// CHECK-SPIRV: ReturnValue [[TWO]]
// CHECK-SPIRV: Label [[CASE_3]]
// CHECK-SPIRV: ReturnValue [[FOUR]]
// CHECK-SPIRV: Label [[CASE_4]]
// CHECK-SPIRV: ReturnValue [[EIGHT]]
// CHECK-SPIRV: Label [[CASE_5]]
// CHECK-SPIRV: ReturnValue [[SIXTEEN]]
// CHECK-SPIRV: FunctionEnd


// CHECK-LLVM: define spir_func i32 @load(i32 addrspace(4)* %[[obj:[0-9a-zA-Z._]+]], i32 %[[order:[0-9a-zA-Z._]+]], i32 %[[scope:[0-9a-zA-Z._]+]]) #0 {
// CHECK-LLVM: entry:
// CHECK-LLVM:  call spir_func i32 @_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(i32 addrspace(4)* %[[obj]], i32 %[[order]], i32 %[[scope]])
// CHECK-LLVM: }
