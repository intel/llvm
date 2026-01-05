; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define i8 @getConstantI8() {
  ret i8 2
}

define i16 @getConstantI16() {
  ret i16 -58
}

define i32 @getConstantI32() {
  ret i32 42
}

define i64 @getConstantI64() {
  ret i64 123456789
}

define i64 @getLargeConstantI64() {
  ret i64 34359738368
}

; CHECK: Capability Int64
; CHECK: Capability Int16
; CHECK: Capability Int8
; CHECK: Name [[#I8_FUNC:]] "getConstantI8"
; CHECK: Name [[#I16_FUNC:]] "getConstantI16"
; CHECK: Name [[#I32_FUNC:]] "getConstantI32"
; CHECK: Name [[#I64_FUNC:]] "getConstantI64"
; CHECK: Name [[#LARGE_I64_FUNC:]] "getLargeConstantI64"

; CHECK: TypeInt [[#I8_TY:]] 8 0
; CHECK: TypeInt [[#I16_TY:]] 16 0
; CHECK: TypeInt [[#I32_TY:]] 32 0
; CHECK: TypeInt [[#I64_TY:]] 64 0
; CHECK: Constant [[#I8_TY]] [[#I8_CONST:]] 2
; CHECK: Constant [[#I16_TY]] [[#I16_CONST:]] 65478
; CHECK: Constant [[#I32_TY]] [[#I32_CONST:]] 42
; CHECK: Constant [[#I64_TY]] [[#I64_CONST:]] 123456789 0
; For 34359738368 = 0x00000008 00000000, so represented as 0 8
; CHECK: Constant [[#I64_TY]] [[#LARGE_I64_CONST:]] 0 8

; CHECK: Function [[#I8_TY]] [[#I8_FUNC]] 0 [[#]]
; CHECK: ReturnValue [[#I8_CONST]]
; CHECK: FunctionEnd

; CHECK: Function [[#I16_TY]] [[#I16_FUNC]] 0 [[#]]
; CHECK: ReturnValue [[#I16_CONST]]
; CHECK: FunctionEnd

; CHECK: Function [[#I32_TY]] [[#I32_FUNC]] 0 [[#]]
; CHECK: ReturnValue [[#I32_CONST]]
; CHECK: FunctionEnd

; CHECK: Function [[#I64_TY]] [[#I64_FUNC]] 0 [[#]]
; CHECK: ReturnValue [[#I64_CONST]]
; CHECK: FunctionEnd

; CHECK: Function [[#I64_TY]] [[#LARGE_I64_FUNC]] 0 [[#]]
; CHECK: ReturnValue [[#LARGE_I64_CONST]]
; CHECK: FunctionEnd

; CHECK-LLVM: ret i8 2 
; CHECK-LLVM: ret i16 -58
; CHECK-LLVM: ret i32 42
; CHECK-LLVM: ret i64 123456789
; CHECK-LLVM: ret i64 34359738368
