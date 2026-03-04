; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

;; Debug info:
; CHECK-SPIRV: Name [[#FOO_ID:]] "foo" 

;; Types:
; CHECK-SPIRV: TypeVoid [[#VOID_TY:]] 
; CHECK-SPIRV: TypeFunction [[#VOID_FUNC_TY:]] [[#VOID_TY]] 

;; Functions:
; CHECK-SPIRV: Function [[#VOID_TY]] [[#FOO_ID]] 0 [[#VOID_FUNC_TY]]
; CHECK-SPIRV-NOT: FunctionParameter
;; NOTE: In 2.4, it isn't explicitly written that a function always has a least
;;       one block. In fact, 2.4.11 seems to imply that there are at least two
;;       blocks in functions with a body, but that doesn't make much sense.
;;       However, in order to distinguish between function declaration and
;;       definition, a function needs at least one block, hence why this test
;;       expects one OpLabel + OpReturn.
; CHECK-SPIRV: Label [[#LBL_ID:]] 
; CHECK-SPIRV: Return 
; CHECK-SPIRV: FunctionEnd

;; Reverse translation checks:
; CHECK-LLVM: define spir_func void @foo()

define void @foo() {
  ret void
}
