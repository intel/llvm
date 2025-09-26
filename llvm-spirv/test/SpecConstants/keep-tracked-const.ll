; This test case ensures that cleaning of temporary constants doesn't purge tracked ones.

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK: TypeInt [[#Int32:]] 32 0
; CHECK: TypeInt [[#Int8:]] 8 0

; CHECK: SpecConstant [[#Int8]] [[#]] 1
; CHECK: Constant [[#Int8]] [[#]] 1
; CHECK: Constant [[#Int8]] [[#]] 0

; CHECK-DAG: Constant [[#Int32]] [[#]] 0
; CHECK-DAG: Constant [[#Int32]] [[#]] 1

; CHECK-LLVM: %conv17.i = sext i8 1 to i64

define spir_kernel void @foo() {
entry:
  %addr = alloca i32
  %r1 = call i8 @_Z20__spirv_SpecConstantia(i32 0, i8 1)
  ; The name '%conv17.i' is important for the test case,
  ; because it includes i32 0 when encoded for SPIR-V usage.
  %conv17.i = sext i8 %r1 to i64
  %tobool = trunc i8 %r1 to i1
  %r2 = zext i1 %tobool to i32
  store i32 %r2, ptr %addr
  ret void
}

declare i8 @_Z20__spirv_SpecConstantia(i32, i8)
