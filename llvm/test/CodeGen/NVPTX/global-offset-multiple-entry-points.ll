; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s
; ModuleID = 'multiple-entry-points.bc'
source_filename = "multiple-entry-points.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This test checks that the pass works with multiple entry points.

declare ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: declare ptr @llvm.nvvm.implicit.offset()

; This function is a kernel entry point that does not use global offset. It will
; not get a clone with a global offset parameter.
; Function Attrs: noinline
define weak_odr dso_local void @_ZTS12third_kernel() {
entry:
  ret void
}

define weak_odr dso_local i64 @_ZTS15common_function() {
; CHECK: define weak_odr dso_local i64 @_ZTS15common_function(ptr %0) {
  %1 = tail call ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: tail call ptr @llvm.nvvm.implicit.offset()
; CHECK: %2 = getelementptr inbounds i32, ptr %0, i64 2
  %2 = getelementptr inbounds i32, ptr %1, i64 2
  %3 = load i32, ptr %2, align 4
  %4 = zext i32 %3 to i64
  ret i64 %4
}

define weak_odr dso_local i64 @_ZTS14first_function() {
; CHECK: define weak_odr dso_local i64 @_ZTS14first_function(ptr %0) {
  %1 = call i64 @_ZTS15common_function()
; CHECK: %2 = call i64 @_ZTS15common_function(ptr %0)
  ret i64 %1
}

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS12first_kernel() {
entry:
; CHECK: %0 = alloca [3 x i32], align 4
; CHECK: call void @llvm.memset.p0.i64(ptr nonnull align 4 dereferenceable(12) %0, i8 0, i64 12, i1 false)
; CHECK: %1 = getelementptr inbounds [3 x i32], ptr %0, i32 0, i32 0
  %0 = call i64 @_ZTS14first_function()
; CHECK: %2 = call i64 @_ZTS14first_function(ptr %1)
  ret void
}

; CHECK: define weak_odr dso_local void @_ZTS12first_kernel_with_offset(ptr byval([3 x i32]) %0) {
; CHECK: entry:
; CHECK:   %1 = call i64 @_ZTS14first_function(ptr %0)
; CHECK:   ret void
; CHECK: }

define weak_odr dso_local i64 @_ZTS15second_function() {
; CHECK: define weak_odr dso_local i64 @_ZTS15second_function(ptr %0) {
  %1 = call i64 @_ZTS15common_function()
; CHECK: %2 = call i64 @_ZTS15common_function(ptr %0)
  ret i64 %1
}

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS13second_kernel() {
entry:
; CHECK: %0 = alloca [3 x i32], align 4
; CHECK: call void @llvm.memset.p0.i64(ptr nonnull align 4 dereferenceable(12) %0, i8 0, i64 12, i1 false)
; CHECK: %1 = getelementptr inbounds [3 x i32], ptr %0, i32 0, i32 0
  %0 = call i64 @_ZTS15second_function()
; CHECK: %2 = call i64 @_ZTS15second_function(ptr %1)
  ret void
}

; CHECK: define weak_odr dso_local void @_ZTS13second_kernel_with_offset(ptr byval([3 x i32]) %0) {
; CHECK: entry:
; CHECK:   %1 = call i64 @_ZTS15second_function(ptr %0)
; CHECK:   ret void
; CHECK: }

; This function doesn't get called by a kernel entry point.
define weak_odr dso_local i64 @_ZTS15no_entry_point() {
; CHECK: define weak_odr dso_local i64 @_ZTS15no_entry_point(ptr %0) {
  %1 = tail call ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: tail call ptr @llvm.nvvm.implicit.offset()
  %2 = getelementptr inbounds i32, ptr %1, i64 2
; CHECK: %2 = getelementptr inbounds i32, ptr %0, i64 2
  %3 = load i32, ptr %2, align 4
  %4 = zext i32 %3 to i64
  ret i64 %4
}

!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3, !5, !6}
; CHECK: !nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3, !5, !6, !7, !8}
!nvvmir.version = !{!9}

!0 = distinct !{ptr @_ZTS12first_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = distinct !{ptr @_ZTS13second_kernel, !"kernel", i32 1}
!6 = distinct !{ptr @_ZTS12third_kernel, !"kernel", i32 1}
; CHECK: !7 = !{ptr @_ZTS13second_kernel_with_offset, !"kernel", i32 1}
; CHECK: !8 = !{ptr @_ZTS12first_kernel_with_offset, !"kernel", i32 1}
!9 = !{i32 1, i32 4}
