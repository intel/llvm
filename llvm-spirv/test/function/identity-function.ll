; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefix=CHECK-LLVM < %t.llc.rev.ll %}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: TypeInt [[#I32_TY:]] 32 0 
; CHECK-SPIRV: TypeFunction [[#FUN_TY:]] [[#I32_TY]] [[#I32_TY]] 

; CHECK-SPIRV: Function [[#I32_TY]] [[#]] 0 [[#FUN_TY]] 
; CHECK-SPIRV: FunctionParameter [[#I32_TY]] [[#]] 
; CHECK-SPIRV: Label [[#LBL:]] 
; CHECK-SPIRV: ReturnValue [[#]] 
; CHECK-SPIRV: FunctionEnd

; CHECK-LLVM: define spir_func i32 @identity(i32 [[value:%.*]])
; CHECK-LLVM:   ret i32 [[value]]

define i32 @identity(i32 %value) {
  ret i32 %value
}
