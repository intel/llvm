; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

;; OpenCL global memory
define ptr addrspace(1) @getConstant1() {
  ret ptr addrspace(1) null
}

;; OpenCL constant memory
define ptr addrspace(2) @getConstant2() {
  ret ptr addrspace(2) null
}

;; OpenCL local memory
define ptr addrspace(3) @getConstant3() {
  ret ptr addrspace(3) null
}

; CHECK: TypeInt [[#I8_TY:]] 8 0 
; CHECK: TypePointer [[#PTR_TY1:]] 5 [[#I8_TY]] 
; CHECK: TypePointer [[#PTR_TY2:]] 0 [[#I8_TY]] 
; CHECK: TypePointer [[#PTR_TY3:]] 4 [[#I8_TY]] 
; CHECK: ConstantNull [[#PTR_TY1]] [[#NULL1:]] 
; CHECK: ConstantNull [[#PTR_TY2]] [[#NULL2:]] 
; CHECK: ConstantNull [[#PTR_TY3]] [[#NULL3:]] 

; CHECK-LLVM: ret ptr addrspace(1) null
; CHECK-LLVM: ret ptr addrspace(2) null
; CHECK-LLVM: ret ptr addrspace(3) null
