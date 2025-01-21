; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; This test checks that, when we fix a deferred type to a known value in the
; type scavenger, we correctly also handle replacing other types that are used
; in the same instruction.

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK: 4 TypeInt [[INT:[0-9]+]] 32 0
; CHECK: 4 TypePointer [[INTPTR:[0-9]+]] 7 [[INT]]

; Function Attrs: nounwind
define spir_kernel void @foo() {
; CHECK: 4 Variable [[INTPTR]] [[IPTR:[0-9]+]] 7
; CHECK: 4 Bitcast [[INTPTR]] [[UPTR:[0-9]+]] [[IPTR]]
; CHECK: 4 Bitcast [[INTPTR]] [[UPTR2:[0-9]+]] [[IPTR]]
entry:
  %iptr = alloca i32, align 4
  %uptr = bitcast ptr %iptr to ptr
  %uptr2 = bitcast ptr %iptr to ptr
  br i1 false, label %a, label %b

a:
; CHECK: 2 Label [[A:[0-9]+]]
  br i1 false, label %c, label %d

b:
; CHECK: 2 Label [[B:[0-9]+]]
  br label %block

c:
; CHECK: 2 Label [[C:[0-9]+]]
  br label %block

d:
; CHECK: 2 Label [[D:[0-9]+]]
  br label %block

block:
; CHECK: 9 Phi [[INTPTR]] {{[0-9]+}} [[UPTR]] [[B]] [[IPTR]] [[C]] [[UPTR2]] [[D]]
  %val = phi ptr [ %uptr, %b ], [ %iptr, %c ], [ %uptr2, %d ]
  ret void
}
