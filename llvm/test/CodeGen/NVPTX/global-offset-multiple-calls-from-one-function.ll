; RUN: opt -enable-new-pm=0 -globaloffset %s -S -o - | FileCheck %s
; ModuleID = 'multiple-calls-from-one-function.bc'
source_filename = "multiple-calls-from-one-function.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda-sycldevice"

; This test checks that when there are multiple calls to a function that uses
; the intrinsic that the caller and the callee only have a single clone each
; with the offset parameter. It also checks that the clone with multiple calls
; to other functions that has a variant that takes an offset parameter will have
; all calls redirected to the corresponding variants.

declare i32* @llvm.nvvm.implicit.offset()
; CHECK-NOT: declare i32* @llvm.nvvm.implicit.offset()

define weak_odr dso_local i64 @_ZTS14other_function() {
; CHECK: define weak_odr dso_local i64 @_ZTS14other_function(i32* %0) {
  %1 = tail call i32* @llvm.nvvm.implicit.offset()
; CHECK-NOT: tail call i32* @llvm.nvvm.implicit.offset()
  %2 = getelementptr inbounds i32, i32* %1, i64 2
; CHECK: %2 = getelementptr inbounds i32, i32* %0, i64 2
  %3 = load i32, i32* %2, align 4
  %4 = zext i32 %3 to i64

  %5 = tail call i32* @llvm.nvvm.implicit.offset()
; CHECK-NOT: tail call i32* @llvm.nvvm.implicit.offset()
  %6 = getelementptr inbounds i32, i32* %5, i64 2
; CHECK: %5 = getelementptr inbounds i32, i32* %0, i64 2
  %7 = load i32, i32* %6, align 4
  %8 = zext i32 %7 to i64

  ret i64 %4
}

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS14example_kernel() {
entry:
; CHECK: %0 = alloca [3 x i32], align 4
; CHECK: %1 = bitcast [3 x i32]* %0 to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* nonnull align 4 dereferenceable(12) %1, i8 0, i64 12, i1 false)
; CHECK: %2 = getelementptr inbounds [3 x i32], [3 x i32]* %0, i32 0, i32 0
  %0 = call i64 @_ZTS14other_function()
; CHECK: %3 = call i64 @_ZTS14other_function(i32* %2)
  %1 = call i64 @_ZTS14other_function()
; CHECK: %4 = call i64 @_ZTS14other_function(i32* %2)
  ret void
}

; CHECK: define weak_odr dso_local void @_ZTS14example_kernel_with_offset([3 x i32]* byval([3 x i32]) %0) {
; CHECK: entry:
; CHECK:   %1 = bitcast [3 x i32]* %0 to i32*
; CHECK:   %2 = call i64 @_ZTS14other_function(i32* %1)
; CHECK:   %3 = call i64 @_ZTS14other_function(i32* %1)
; CHECK:   ret void
; CHECK: }

!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
; CHECK: !nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3, !5}
!nvvmir.version = !{!6}

!0 = distinct !{void ()* @_ZTS14example_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
; CHECK: !5 = !{void ([3 x i32]*)* @_ZTS14example_kernel_with_offset, !"kernel", i32 1}
!6 = !{i32 1, i32 4}
