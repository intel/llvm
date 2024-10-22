; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_KHR_untyped_pointers

; RUN: spirv-val %t.spv

; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability UntypedPointersKHR
; CHECK-SPIRV: Extension "SPV_KHR_untyped_pointers"
; CHECK-SPIRV: TypeUntypedPointerKHR [[#UntypedPtrTy:]] 7
; CHECK-SPIRV: TypeFunction [[#FuncTy:]] [[#UntypedPtrTy]] [[#UntypedPtrTy]]

; CHECK-SPIRV: Function [[#UntypedPtrTy]] [[#ProcessFuncId:]] 0 [[#FuncTy]]
; CHECK-SPIRV: FunctionParameter [[#UntypedPtrTy]]

; CHECK-SPIRV: Function [[#UntypedPtrTy]] [[#FuncId:]] 0 [[#FuncTy]]
; CHECK-SPIRV: FunctionParameter [[#UntypedPtrTy]] [[#ParamId:]]
; CHECK-SPIRV: FunctionCall [[#UntypedPtrTy]] [[#Res:]] [[#ProcessFuncId]] [[#ParamId]]
; CHECK-SPIRV: ReturnValue [[#Res]]


; CHECK-LLVM: declare spir_func ptr @processPointer(ptr)
; CHECK-LLVM: define spir_func ptr @example(ptr %arg)
; CHECK-LLVM: entry:
; CHECK-LLVM:   %result = call spir_func ptr @processPointer(ptr %arg)
; CHECK-LLVM:   ret ptr %result

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-G1"
target triple = "spir64-unknown-unknown"

declare ptr @processPointer(ptr)

define ptr @example(ptr %arg) {
entry:
	%result = call ptr @processPointer(ptr %arg)
	ret ptr %result
}
