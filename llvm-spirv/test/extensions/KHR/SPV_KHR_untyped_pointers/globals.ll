; This test validated untyped access chain and its use in SpecConstantOp.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -o %t.spv
; TODO: enable back once spirv-tools are updated and allow untyped access chain as OpSpecConstantOp operand.
; R/UN: spirv-val %t.spv
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV-DAG: TypeInt [[#I16:]] 16 0
; CHECK-SPIRV-DAG: TypeInt [[#I32:]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[#I64:]] 64 0
; CHECK-SPIRV-DAG: Constant [[#I16]] [[#CONST0:]] 0
; CHECK-SPIRV-DAG: Constant [[#I32]] [[#CONST2:]] 2
; CHECK-SPIRV-DAG: Constant [[#I32]] [[#CONST3:]] 3
; CHECK-SPIRV-DAG: Constant [[#I32]] [[#CONST4:]] 4
; CHECK-SPIRV-DAG: Constant [[#I64]] [[#CONST0_I64:]] 0
; CHECK-SPIRV-DAG: Constant [[#I64]] [[#CONST1_I64:]] 1
; CHECK-SPIRV-DAG: Constant [[#I64]] [[#CONST2_I64:]] 2
; CHECK-SPIRV-DAG: Constant [[#I64]] [[#CONST3_I64:]] 3

; CHECK-SPIRV-DAG: TypeUntypedPointerKHR [[#PTRTY:]] 5
; CHECK-SPIRV-DAG: TypeUntypedPointerKHR [[#LOCALPTRTY:]] 4
; CHECK-SPIRV-DAG: TypeArray [[#ARRAYTY:]] [[#PTRTY]] [[#CONST2]]
; CHECK-SPIRV-DAG: TypePointer [[#ARRAYPTRTY:]] 5 [[#ARRAYTY]]
; CHECK-SPIRV-DAG: TypeArray [[#ARRAY1TY:]] [[#I32]] [[#CONST4]]
; CHECK-SPIRV-DAG: TypeArray [[#ARRAY2TY:]] [[#ARRAY1TY]] [[#CONST3]]
; CHECK-SPIRV-DAG: TypeArray [[#ARRAY3TY:]] [[#ARRAY2TY]] [[#CONST2]]
; CHECK-SPIRV-DAG: TypePointer [[#ARRAY3PTRTY:]] 5 [[#ARRAY3TY]]


; CHECK-SPIRV: UntypedVariableKHR [[#PTRTY]] [[#VARA:]] 5 [[#I16]] [[#CONST0]]
; CHECK-SPIRV: UntypedVariableKHR [[#PTRTY]] [[#VARB:]] 5 [[#I32]]
; CHECK-SPIRV: UntypedVariableKHR [[#PTRTY]] [[#VARC:]] 5 [[#PTRTY]] [[#VARA]]
; CHECK-SPIRV: UntypedVariableKHR [[#LOCALPTRTY]] [[#VARD:]] 4 [[#PTRTY]]
; CHECK-SPIRV: Variable [[#ARRAYPTRTY]] [[#VARE:]] 5
; CHECK-SPIRV: Variable [[#ARRAY3PTRTY]] [[#VARF:]] 5
; CHECK-SPIRV: SpecConstantOp [[#PTRTY]] [[#SPECCONST:]] 4424 [[#ARRAY3TY]] [[#VARF]] [[#CONST0_I64]] [[#CONST1_I64]] [[#CONST2_I64]] [[#CONST3_I64]]
; CHECK-SPIRV: UntypedVariableKHR [[#PTRTY]] [[#VARG:]] 5 [[#PTRTY]] [[#SPECCONST]]

; CHECK-LLVM: @a = addrspace(1) global i16 0
; CHECK-LLVM: @b = external addrspace(1) global i32
; CHECK-LLVM: @c = addrspace(1) global ptr addrspace(1) @a
; CHECK-LLVM: @d = external addrspace(3) global ptr addrspace(1)
; CHECK-LLVM: @e = addrspace(1) global [2 x ptr addrspace(1)] [ptr addrspace(1) @a, ptr addrspace(1) @b]
; CHECK-LLVM: @f = addrspace(1) global [2 x [3 x [4 x i32]]]
; CHECK-LLVM: @g = addrspace(1) global ptr addrspace(1) getelementptr inbounds ([2 x [3 x [4 x i32]]], ptr addrspace(1) @f, i64 0, i64 1, i64 2, i64 3)

@a = addrspace(1) global i16 0
@b = external addrspace(1) global i32
@c = addrspace(1) global ptr addrspace(1) @a
@d = external addrspace(3) global ptr addrspace(1)
@e = addrspace(1) global [2 x ptr addrspace(1)] [ptr addrspace(1) @a, ptr addrspace(1) @b]
@f = addrspace(1) global [2 x [3 x [4 x i32]]] [[3 x [4 x i32]] [[4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32] [i32 1, i32 2, i32 3, i32 4]], [3 x [4 x i32]] [[4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32] [i32 1, i32 2, i32 3, i32 4]]]
@g = addrspace(1) global ptr addrspace(1) getelementptr inbounds ([2 x [3 x [4 x i32]]], ptr addrspace(1) @f, i64 0, i64 1, i64 2, i64 3)

; Function Attrs: nounwind
define spir_func void @foo() {
entry:
  ret void
}
