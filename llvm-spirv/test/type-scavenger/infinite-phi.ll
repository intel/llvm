; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK: 4 TypeInt [[INT:[0-9]+]] 32 0
; CHECK: 4 TypePointer [[INTPTR:[0-9]+]] 7 [[INT]]
; CHECK: 3 TypeFloat [[FLOAT:[0-9]+]] 32
; CHECK: 4 TypePointer [[FLOATPTR:[0-9]+]] 7 [[FLOAT]]

; Function Attrs: nounwind
define spir_kernel void @foo() {
; CHECK: 4 Variable [[INTPTR]] [[IPTR:[0-9]+]] 7
; CHECK: 4 Variable [[FLOATPTR]] [[FPTR:[0-9]+]] 7
; CHECK: 4 Bitcast [[FLOATPTR]] [[IPTRI:[0-9]+]] [[IPTR]]
; CHECK: 7 Phi [[FLOATPTR]] [[PTR1:[0-9]+]] [[PTR2:[0-9]+]] {{[0-9]+}} [[IPTRI]]
; CHECK: 7 Phi [[FLOATPTR]] [[PTR2]] [[PTR1]] {{[0-9]+}} [[FPTR]]
entry:
  %iptr = alloca i32, align 4
  %fptr = alloca float, align 4
  br label %loop

loop:
  %ptr1 = phi ptr [%ptr2, %loop], [%iptr, %entry]
  %ptr2 = phi ptr [%ptr1, %loop], [%fptr, %entry]
  %cond = phi i32 [0, %entry], [%cond.next, %loop]
  %cond.next = add i32 %cond, 1
  %cmp = icmp slt i32 %cond.next, 150
  br i1 %cmp, label %exit, label %loop

exit:
  ret void
}
