; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s
; ModuleID = 'intrinsic-function-mix.bc'
source_filename = "global-offset-intrinsic-function-mix.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This test checks that the pass works with functions containing calls to the
; intrinsic and calls to functions that call the intrinsic

declare ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: declare ptr @llvm.nvvm.implicit.offset()

define weak_odr dso_local i64 @_ZTS15other_function() {
; CHECK: define weak_odr dso_local i64 @_ZTS15other_function() {
  %1 = tail call ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: tail call ptr @llvm.nvvm.implicit.offset()
  %2 = getelementptr inbounds i32, ptr %1, i64 2
  %3 = load i32, ptr %2, align 4
  %4 = zext i32 %3 to i64
; CHECK: %1 = zext i32 0 to i64
  ret i64 %4
}

; CHECK: define weak_odr dso_local i64 @_ZTS15other_function_with_offset(ptr %0) {  
; CHECK: %2 = getelementptr inbounds i32, ptr %0, i64 2
; CHECK: %3 = load i32, ptr %2, align 4
; CHECK: %4 = zext i32 %3 to i64
; CHECK: ret i64 %4
; CHECK: }

define weak_odr dso_local i64 @_ZTS14mixed_function() {
; CHECK: define weak_odr dso_local i64 @_ZTS14mixed_function() {
  %1 = call i64 @_ZTS15other_function()
; CHECK: %1 = call i64 @_ZTS15other_function()
  %2 = tail call ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: tail call ptr @llvm.nvvm.implicit.offset()
  %3 = getelementptr inbounds i32, ptr %2, i64 2
  %4 = load i32, ptr %3, align 4
  %5 = zext i32 %4 to i64
; CHECK: %2 = zext i32 0 to i64
  ret i64 %1
}

; CHECK: define weak_odr dso_local i64 @_ZTS14mixed_function_with_offset(ptr %0) {
; CHECK: %2 = call i64 @_ZTS15other_function_with_offset(ptr %0)
; CHECK: %3 = getelementptr inbounds i32, ptr %0, i64 2
; CHECK: %4 = load i32, ptr %3, align 4
; CHECK: %5 = zext i32 %4 to i64
; CHECK: ret i64 %2

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS12some_kernel() {
entry:
  %0 = call i64 @_ZTS14mixed_function()
; CHECK: %0 = call i64 @_ZTS14mixed_function()
  ret void
}

; CHECK: define weak_odr dso_local void @_ZTS12some_kernel_with_offset(ptr byval([3 x i32]) %0) {
; CHECK: entry:
; CHECK:   %1 = call i64 @_ZTS14mixed_function_with_offset(ptr %0)
; CHECK:   ret void
; CHECK: }

!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3, !5}
!nvvmir.version = !{!5}

!0 = distinct !{ptr @_ZTS12some_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = !{i32 1, i32 4}
; CHECK: !6 = !{ptr @_ZTS12some_kernel_with_offset, !"kernel", i32 1}
