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

; Check there is an entrypoint and a function produced.
; CHECK-SPIRV: EntryPoint 6 [[EP:[0-9]+]] "testfunction"
; CHECK-SPIRV: Name [[FUNC:[0-9]+]] "testfunction"
; CHECK-SPIRV: Decorate [[FUNC]] LinkageAttributes "testfunction" Export
; CHECK-SPIRV: Function 2 [[FUNC]] 0 3
; CHECK-SPIRV: Function 2 [[EP]] 0 3
; CHECK-SPIRV: FunctionCall 2 8 [[FUNC]]