; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This test checks that the transformation is applied in the basic case.

declare ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: llvm.nvvm.implicit.offset

define i64 @_ZTS14other_function() {
; CHECK: define i64 @_ZTS14other_function() {
  %1 = tail call ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: tail call ptr @llvm.nvvm.implicit.offset()
  %2 = getelementptr inbounds i32, ptr %1, i64 2
  %3 = load i32, ptr %2, align 4
  %4 = zext i32 %3 to i64
  ret i64 %4
}

; CHECK: define i64 @_ZTS14other_function_with_offset(ptr %0) {
; CHECK-NOT: tail call ptr @llvm.nvvm.implicit.offset()
; CHECK: %2 = getelementptr inbounds i32, ptr %0, i64 2
; CHECK: %3 = load i32, ptr %2, align 4
; CHECK: %4 = zext i32 %3 to i64
; CHECK: ret i64 %4
; CHECK: }

define ptx_kernel void @_ZTS14example_kernel() {
entry:
  %0 = call i64 @_ZTS14other_function()
; CHECK: %0 = call i64 @_ZTS14other_function()
  ret void
}

; CHECK: define ptx_kernel void @_ZTS14example_kernel_with_offset(ptr byval([3 x i32]) %0) {
; CHECK: entry:
; CHECK:   %1 = call i64 @_ZTS14other_function_with_offset(ptr %0)
; CHECK:   ret void
; CHECK: }

; CHECK: !nvvm.annotations = {{.*}}

!llvm.module.flags = !{!7}
!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
!nvvmir.version = !{!6}

!0 = distinct !{ptr @_ZTS14example_kernel, !"dummy", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!6 = !{i32 1, i32 4}
!7 = !{i32 1, !"sycl-device", i32 1}
