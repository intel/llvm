; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV: TypeInt [[#INT:]] 32 0
; CHECK-SPIRV: TypeUntypedPointerKHR [[#PTRTY:]] 7
; CHECK-SPIRV: TypeFloat [[#FLOAT:]] 32

; CHECK-SPIRV: UntypedVariableKHR [[#PTRTY]] [[#IPTR:]] 7 [[#INT]]
; CHECK-SPIRV: UntypedVariableKHR [[#PTRTY]] [[#FPTR:]] 7 [[#FLOAT]]

; This bitcast seems redundant, it appears because of the Type Scavenger.
; TODO: Investigate and fix if possible.
; CHECK-SPIRV: Bitcast [[#PTRTY]] [[#IPTRI:]] [[#IPTR]]
; CHECK-SPIRV: Phi [[#PTRTY]] [[#PTR1:]] [[#PTR2:]] [[#]] [[#IPTRI]]
; CHECK-SPIRV: Phi [[#PTRTY]] [[#PTR2]] [[#PTR1]] [[#]] [[#FPTR]]

; CHECK-LLVM: %iptr = alloca i32, align 4
; CHECK-LLVM: %fptr = alloca float, align 4
; CHECK-LLVM: %0 = bitcast ptr %iptr to ptr
; CHECK-LLVM: br label %loop

; CHECK-LLVM: %ptr1 = phi ptr [ %ptr2, %loop ], [ %0, %entry ]
; CHECK-LLVM: %ptr2 = phi ptr [ %ptr1, %loop ], [ %fptr, %entry ]
; CHECK-LLVM: %cond = phi i32 [ 0, %entry ], [ %cond.next, %loop ]
; CHECK-LLVM: %cond.next = add i32 %cond, 1
; CHECK-LLVM: %cmp = icmp slt i32 %cond.next, 150
; CHECK-LLVM: br i1 %cmp, label %exit, label %loop

; Function Attrs: nounwind
define spir_kernel void @foo() {
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
