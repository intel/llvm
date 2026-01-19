; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define half @getConstantFP16() {
  ret half 0x3ff1340000000000 ; 0x3c4d represented as double.
}

define float @getConstantFP32() {
  ret float 0x3fd27c8be0000000 ; 0x3e93e45f represented as double
}

define double @getConstantFP64() {
  ret double 0x4f2de42b8c68f3f1
}

; CHECK: Capability Float16Buffer
; CHECK: Capability Float64
; CHECK: Name [[#FUNC_FP16:]] "getConstantFP16"
; CHECK: Name [[#FUNC_FP32:]] "getConstantFP32"
; CHECK: Name [[#FUNC_FP64:]] "getConstantFP64"
; CHECK: TypeFloat [[#FP16_TY:]] 16
; CHECK: TypeFloat [[#FP32_TY:]] 32
; CHECK: TypeFloat [[#FP64_TY:]] 64

; CHECK: Constant [[#FP16_TY]] [[#FP16_CONST:]] 15437
; CHECK: Constant [[#FP32_TY]] [[#FP32_CONST:]] 1049879647
; CHECK: Constant [[#FP64_TY]] [[#FP64_CONST:]] 2355688433 1328407595

; CHECK: Function [[#FP16_TY]] [[#FUNC_FP16]] 0 [[#]]
; CHECK: ReturnValue [[#FP16_CONST]]
; CHECK: FunctionEnd

; CHECK: Function [[#FP32_TY]] [[#FUNC_FP32]] 0 [[#]]
; CHECK: ReturnValue [[#FP32_CONST]]
; CHECK: FunctionEnd

; CHECK: Function [[#FP64_TY]] [[#FUNC_FP64]] 0 [[#]]
; CHECK: ReturnValue [[#FP64_CONST]]
; CHECK: FunctionEnd

; CHECK-LLVM: ret half 0xH3C4D
; CHECK-LLVM: ret float 0x3FD27C8BE0000000
; CHECK-LLVM: ret double 0x4F2DE42B8C68F3F1
