;; Test to check that freeze instruction does not cause a crash
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; All freeze instructions should be deleted and uses of freeze's result should be replaced
; with freeze's source or a random constant if freeze's source is poison or undef.
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM --implicit-check-not="= freeze"

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; test i32
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK-LLVM: @testfunction_i32A
; Uses of result should be replaced with freeze's source
; CHECK-LLVM-NEXT: add nsw i32 %val, 1
define spir_func i32 @testfunction_i32A(i32 %val) {
   %1 = freeze i32 %val
   %2 = add nsw i32 %1, 1
   ret i32 %2
}

; CHECK-LLVM: @testfunction_i32B
; Frozen poison/undef should produce a constant.
; add should be deleted since both inputs are constant.
; CHECK-LLVM-NEXT: ret i32
define spir_func i32 @testfunction_i32B(i32 %val) {
   %1 = freeze i32 poison
   %2 = add nsw i32 %1, 1   
   ret i32 %2
}

; CHECK-LLVM: @testfunction_i32C
; Frozen poison/undef should produce a constant.
; add should be deleted since both inputs are constant.
; CHECK-LLVM-NEXT: ret i32
define spir_func i32 @testfunction_i32C(i32 %val) {
   %1 = freeze i32 undef
   %2 = add nsw i32 %1, 1   
   ret i32 %2
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; test float
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK-LLVM: @testfunction_floatA
; freeze should be eliminated.
; Uses of result should be replaced with freeze's source
; CHECK-LLVM-NEXT: fadd float %val
define spir_func float @testfunction_floatA(float %val) {
   %1 = freeze float %val
   %2 = fadd float %1, 1.0
   ret float %2
}

; CHECK-LLVM: @testfunction_floatB
; Frozen poison/undef should produce a constant.
; add should be deleted since both inputs are constant.
; CHECK-LLVM-NEXT: ret float
define spir_func float @testfunction_floatB(float %val) {
   %1 = freeze float poison
   %2 = fadd float %1, 1.0
   ret float %2
}

; CHECK-LLVM: @testfunction_floatC
; Frozen poison/undef should produce a constant.
; add should be deleted since both inputs are constant.
; CHECK-LLVM-NEXT: ret float
define spir_func float @testfunction_floatC(float %val) {
   %1 = freeze float undef
   %2 = fadd float %1, 1.0
   ret float %2
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; test ptr
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK-LLVM: @testfunction_ptrA
; freeze should be eliminated.
; Uses of result should be replaced with freeze's source
; CHECK-LLVM-NEXT: ptrtoint ptr %val to i64
define spir_func i64 @testfunction_ptrA(ptr %val) {
   %1 = freeze ptr %val
   %2 = ptrtoint ptr %1 to i64
   ret i64 %2
}

; CHECK-LLVM: @testfunction_ptrB
; Frozen poison/undef should produce a constant.
; For ptr type this constant is null.
; CHECK-LLVM-NEXT: ptrtoint ptr null to i64
define spir_func i64 @testfunction_ptrB(ptr addrspace(1) %val) {
   %1 = freeze ptr poison
   %2 = ptrtoint ptr %1 to i64
   ret i64 %2
}

; CHECK-LLVM: @testfunction_ptrC
; Frozen poison/undef should produce a constant.
; For ptr type this constant is null.
; CHECK-LLVM-NEXT: ptrtoint ptr null to i64
define spir_func i64 @testfunction_ptrC(ptr addrspace(1) %val) {
   %1 = freeze ptr undef
   %2 = ptrtoint ptr %1 to i64
   ret i64 %2
}
