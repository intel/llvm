; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s
; ModuleID = 'multiple-calls-from-one-function.bc'
source_filename = "multiple-calls-from-one-function.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This test checks that when there are multiple calls to a function that uses
; the intrinsic that the caller and the callee have two clones each - one with
; the offset parameter and one without. It also checks that a clone with
; multiple calls to other functions that have a clone as well will have
; all calls redirected to the corresponding variants.

declare ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: declare ptr @llvm.nvvm.implicit.offset()

define weak_odr dso_local i64 @_ZTS14other_function() {
; CHECK: define weak_odr dso_local i64 @_ZTS14other_function() {
  %1 = tail call ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: tail call ptr @llvm.nvvm.implicit.offset()
  %2 = getelementptr inbounds i32, ptr %1, i64 2
  %3 = load i32, ptr %2, align 4
  %4 = zext i32 %3 to i64
; CHECK %1 = zext i32 0 to i64

  %5 = tail call ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: tail call ptr @llvm.nvvm.implicit.offset()
  %6 = getelementptr inbounds i32, ptr %5, i64 2
  %7 = load i32, ptr %6, align 4
  %8 = zext i32 %7 to i64
; CHECK: %2 = zext i32 0 to i64

  ret i64 %4
; CHECK: ret i64 %1
}

; CHECK: define weak_odr dso_local i64 @_ZTS14other_function_with_offset(ptr %0) {
; CHECK: %2 = getelementptr inbounds i32, ptr %0, i64 2
; CHECK: %3 = load i32, ptr %2, align 4
; CHECK: %4 = zext i32 %3 to i64
; CHECK: %5 = getelementptr inbounds i32, ptr %0, i64 2
; CHECK: %6 = load i32, ptr %5, align 4
; CHECK: %7 = zext i32 %6 to i64
; CHECK: ret i64 %4
; CHECK: }

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS14example_kernel() {
entry:
  %0 = call i64 @_ZTS14other_function()
; CHECK: %0 = call i64 @_ZTS14other_function()
  %1 = call i64 @_ZTS14other_function()
; CHECK: %1 = call i64 @_ZTS14other_function()
  ret void
}

; CHECK: define weak_odr dso_local void @_ZTS14example_kernel_with_offset(ptr byval([3 x i32]) %0) {
; CHECK: entry:
; CHECK:   %1 = call i64 @_ZTS14other_function_with_offset(ptr %0)
; CHECK:   %2 = call i64 @_ZTS14other_function_with_offset(ptr %0)
; CHECK:   ret void
; CHECK: }

!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
; CHECK: !nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3, !5}
!nvvmir.version = !{!6}

!0 = distinct !{ptr @_ZTS14example_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
; CHECK: !5 = !{ptr @_ZTS14example_kernel_with_offset, !"kernel", i32 1}
!6 = !{i32 1, i32 4}
