;; Test to check that an LLVM spir_kernel gets translated into an
;; Entrypoint wrapper and Function with LinkageAttributes
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_kernel void @testfunction() {
   ret void
}

define spir_kernel void @callerfunction() {
   call spir_kernel void @testfunction()
   call spir_kernel void @testdeclaration()
   ret void
}

declare spir_kernel void @testdeclaration()

; Check there is an entrypoint and a function produced.
; CHECK-SPIRV: EntryPoint 6 [[#TestEn:]] "testfunction"
; CHECK-SPIRV: EntryPoint 6 [[#CallerEn:]] "callerfunction"
; CHECK-SPIRV: Name [[#TestDecl:]] "testdeclaration"
; CHECK-SPIRV: Name [[#TestFn:]] "testfunction"
; CHECK-SPIRV: Name [[#CallerFn:]] "callerfunction"
; CHECK-SPIRV: Decorate [[#TestDecl]] LinkageAttributes "testdeclaration" Import
; CHECK-SPIRV: Decorate [[#TestFn]] LinkageAttributes "testfunction" Export
; CHECK-SPIRV: Decorate [[#CallerFn]] LinkageAttributes "callerfunction" Export

; CHECK-SPIRV: Function [[#]] [[#TestDecl]] [[#]] [[#]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: FunctionEnd

; CHECK-SPIRV: Function [[#]] [[#TestFn]] [[#]] [[#]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: Return
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: FunctionEnd

; CHECK-SPIRV: Function [[#]] [[#CallerFn]] [[#]] [[#]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: FunctionCall [[#]] [[#]] [[#TestFn]]
; CHECK-SPIRV-NEXT: FunctionCall [[#]] [[#]] [[#TestDecl]]
; CHECK-SPIRV-NEXT: Return
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: FunctionEnd


; CHECK-SPIRV: Function [[#]] [[#TestEn]] [[#]] [[#]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: FunctionCall [[#]] [[#]] [[#TestFn]]
; CHECK-SPIRV-NEXT: Return
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: FunctionEnd

; CHECK-SPIRV: Function [[#]] [[#CallerEn]] [[#]] [[#]]
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: Label
; CHECK-SPIRV-NEXT: FunctionCall [[#]] [[#]] [[#CallerFn]]
; CHECK-SPIRV-NEXT: Return
; CHECK-SPIRV-EMPTY:
; CHECK-SPIRV-NEXT: FunctionEnd
