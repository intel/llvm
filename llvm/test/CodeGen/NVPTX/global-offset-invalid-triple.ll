; RUN: not --crash llc -march=nvptx64 -mcpu=sm_20 %s -o - 2>&1 | FileCheck %s
; ModuleID = 'invalid-triple.bc'
; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.implicit.offset
source_filename = "invalid-triple.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-nvcl"

; This test checks that the pass does not run on nvcl triples.

declare ptr @llvm.nvvm.implicit.offset()

define i64 @_ZTS14other_function() {
  %1 = tail call ptr @llvm.nvvm.implicit.offset()
  %2 = getelementptr inbounds i32, ptr %1, i64 2
  %3 = load i32, ptr %2, align 4
  %4 = zext i32 %3 to i64
  ret i64 %4
}

define void @_ZTS14example_kernel() {
entry:
  %0 = call i64 @_ZTS14other_function()
  ret void
}

!llvm.module.flags = !{!6}
!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
!nvvmir.version = !{!5}

!0 = distinct !{ptr @_ZTS14example_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = !{i32 1, i32 4}
!6 = !{i32 1, !"sycl-device", i32 1}
