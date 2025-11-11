; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Name [[#BAR_ID:]] "bar" 
; CHECK-SPIRV: Name [[#FOO_ID:]] "foo" 
; CHECK-SPIRV: TypeInt [[#I32_TY:]] 32 0 
; CHECK-SPIRV: TypeFunction [[#BAR_FUNC_TY:]] [[#I32_TY]] [[#I32_TY]] 
; CHECK-SPIRV: TypeVoid [[#VOID_TY:]] 
; CHECK-SPIRV: TypeFunction [[#FOO_FUNC_TY:]] [[#VOID_TY]] [[#I32_TY]] 
;; Function decl:
; CHECK-SPIRV: Function [[#I32_TY]] [[#BAR_ID]] 0 [[#BAR_FUNC_TY]] 
; CHECK-SPIRV: FunctionParameter [[#I32_TY]] [[#]] 
; CHECK-SPIRV: FunctionEnd 

; CHECK-SPIRV: Function [[#VOID_TY]] [[#FOO_ID]] 0 [[#FOO_FUNC_TY]] 
; CHECK-SPIRV: FunctionParameter [[#I32_TY]] [[#X_ID:]] 
; CHECK-SPIRV: Label [[#LBL:]] 
; CHECK-SPIRV: FunctionCall [[#I32_TY]] [[#]] [[#BAR_ID]] [[#X_ID]] 
; CHECK-SPIRV: Return 
; CHECK-SPIRV-NOT: Label
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: declare spir_func i32 @bar(i32)
; CHECK-LLVM: define spir_func void @foo(i32 [[x:%.*]])
; CHECK-LLVM:   [[call1:%.*]] = call spir_func i32 @bar(i32 [[x]])

declare i32 @bar(i32 %x)

define spir_func void @foo(i32 %x) {
  %call1 = call spir_func i32 @bar(i32 %x)
  ret void
}
