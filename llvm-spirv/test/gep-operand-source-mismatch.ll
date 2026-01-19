; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck %s --input-file %t.spt -check-prefix=CHECK-SPIRV
; RUN: FileCheck %s --input-file %t.rev.ll  -check-prefix=CHECK-LLVM

; Make sure that when the GEP operand type doesn't match the source element type (here operand a_var is [2 x i16], but the source element is i8),
; we cast the operand to the source element pointer type (a_var to i8*).

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV-DAG: Name [[A_VAR:[0-9]+]] "a_var"
; CHECK-SPIRV-DAG: Name [[GLOBAL_PTR:[0-9]+]] "global_ptr"

; CHECK-SPIRV-DAG: TypeArray [[ARRAY_TYPE:[0-9]+]] [[USHORT_TYPE:[0-9]+]] [[CONST_2:[0-9]+]]
; CHECK-SPIRV-DAG: TypePointer [[ARRAY_PTR_TYPE:[0-9]+]] 5 [[ARRAY_TYPE]]

; CHECK-SPIRV-DAG: Variable [[ARRAY_PTR_TYPE]] [[A_VAR]] 5 [[INIT_ID:[0-9]+]]
; CHECK-SPIRV-DAG: SpecConstantOp [[I8PTR:[0-9]+]] [[BITCAST:[0-9]+]] 124 [[A_VAR]]
; CHECK-SPIRV-DAG: SpecConstantOp [[I8PTR]] [[PTRCHAIN:[0-9]+]] 67 [[BITCAST]] [[INDEX_ID:[0-9]+]]
; CHECK-SPIRV-DAG: TypePointer [[PTR_PTR_TYPE:[0-9]+]] 5 [[I8PTR]]
; CHECK-SPIRV-DAG: Variable [[PTR_PTR_TYPE]] [[GLOBAL_PTR]] 5 [[PTRCHAIN]]

; CHECK-LLVM: @global_ptr = addrspace(1) global ptr addrspace(1) getelementptr (i8, ptr addrspace(1) @a_var, i64 2), align 8
; CHECK-LLVM-NOT: @global_ptr = addrspace(1) global ptr addrspace(1) getelementptr ([2 x i16], ptr addrspace(1) @a_var, i64 2), align 8

@a_var = dso_local addrspace(1) global [2 x i16] [i16 4, i16 5], align 2
@global_ptr = dso_local addrspace(1) global ptr addrspace(1) getelementptr (i8, ptr addrspace(1) @a_var, i64 2), align 8
