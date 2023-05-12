; RUN: opt -opaque-pointers < %s -passes=LowerESIMD -S | FileCheck %s

; This test checks we lower vector the SPIRV global offset intrinsic with opaque pointers

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@__spirv_BuiltInGlobalOffset = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

define spir_kernel void @foo() {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP:%.*]] = alloca i64, align 8
; CHECK-NEXT:    [[ADD:%.*]] = add i64 0, 5
; CHECK-NEXT:   store i64 [[ADD]], ptr [[TMP]], align 8

; Verify that the attribute is deleted from GenX declaration
; CHECK-NOT: readnone
entry:
  %tmp = alloca i64, align 8
  %0 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalOffset, align 32
  %1 = add i64 %0, 5
  store i64 %1, ptr %tmp, align 8
 ret void
}
