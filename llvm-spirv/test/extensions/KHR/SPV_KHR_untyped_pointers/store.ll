; This test checks translation of function parameter which is untyped pointer.
; Lately, when we do support untyped variables, this one could be used to check
; "full" forward and reverse translation of opaque pointers.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -o %t.spv
; RUN: spirv-val %t.spv

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability UntypedPointersKHR
; CHECK-SPIRV: Extension "SPV_KHR_untyped_pointers"
; CHECK-SPIRV-DAG: TypeInt [[#IntTy:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#IntTy]] [[#Constant0:]] 0
; CHECK-SPIRV-DAG: Constant [[#IntTy]] [[#Constant42:]] 42
; CHECK-SPIRV-DAG: TypeUntypedPointerKHR [[#UntypedPtrTy:]] 5
; CHECK-SPIRV-DAG: TypeUntypedPointerKHR [[#UntypedPtrTyFunc:]] 7

; CHECK-SPIRV: FunctionParameter [[#UntypedPtrTy]] [[#FuncParam:]]
; CHECK-SPIRV: UntypedVariableKHR [[#UntypedPtrTyFunc]] [[#VarBId:]] 7 [[#UntypedPtrTy]]
; CHECK-SPIRV: Store [[#VarBId]] [[#FuncParam]] 2 4
; CHECK-SPIRV: Load [[#UntypedPtrTy]] [[#LoadId:]] [[#VarBId]] 2 4
; CHECK-SPIRV: Store [[#LoadId]] [[#Constant0]] 2 4

; CHECK-SPIRV: FunctionParameter [[#UntypedPtrTy]] [[#FuncParam0:]]
; CHECK-SPIRV: FunctionParameter [[#UntypedPtrTy]] [[#FuncParam1:]]
; CHECK-SPIRV: UntypedVariableKHR [[#UntypedPtrTyFunc]] [[#VarCId:]] 7 [[#IntTy]]
; CHECK-SPIRV: Store [[#VarCId]] [[#Constant42]] 2 4
; CHECK-SPIRV: Load [[#IntTy]] [[#LoadId:]] [[#FuncParam1]] 2 4
; CHECK-SPIRV: Store [[#FuncParam0]] [[#LoadId]] 2 4

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-LLVM: define spir_func void @foo(ptr addrspace(1) %a)
; CHECK-LLVM:   %b = alloca ptr addrspace(1), align 4
; CHECK-LLVM:   store ptr addrspace(1) %a, ptr %b, align 4
; CHECK-LLVM:   %0 = load ptr addrspace(1), ptr %b, align 4
; CHECK-LLVM:   store i32 0, ptr addrspace(1) %0, align 4
define spir_func void @foo(ptr addrspace(1) %a) {
entry:
  %b = alloca ptr addrspace(1), align 4
  store ptr addrspace(1) %a, ptr %b, align 4
  %0 = load ptr addrspace(1), ptr %b, align 4
  store i32 0, ptr addrspace(1) %0, align 4
  ret void
}

; CHECK-LLVM: define spir_func void @boo(ptr addrspace(1) %0, ptr addrspace(1) %1)
; CHECK-LLVM: %c = alloca i32
; CHECK-LLVM: store i32 42, ptr %c, align 4
; CHECK-LLVM: %2 = load i32, ptr addrspace(1) %1, align 4
; CHECK-LLVM: store i32 %2, ptr addrspace(1) %0, align 4
define dso_local void @boo(ptr addrspace(1) %0, ptr addrspace(1) %1) {
entry:
  %c = alloca i32, align 4
  store i32 42, ptr %c, align 4
  %2 = load i32, ptr addrspace(1) %1, align 4
  store i32 %2, ptr addrspace(1) %0, align 4
  ret void
}
