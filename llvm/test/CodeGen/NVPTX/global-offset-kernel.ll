; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This test checks that the transformation is applied in the basic case
; that __spirv_BuiltInGlobalOffset is used in kernel.

declare i64 @_Z27__spirv_BuiltInGlobalOffseti(i32)

declare void @use(i64)

define ptx_kernel void @_ZTS14example_kernel() {
; CHECK-LABEL: define ptx_kernel void @_ZTS14example_kernel() {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    call void @use(i64 0)
; CHECK-NEXT:    ret void
;
entry:
  %0 = call i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 2)
  call void @use(i64 %0)
  ret void
}

; CHECK-LABEL: define ptx_kernel void @_ZTS14example_kernel_with_offset(
; CHECK-SAME:    ptr byval([3 x i32]) [[A:%.*]])
; CHECK-NEXT: [[ENTRY:.*:]]
; CHECK-NEXT:   [[GEP:%.*]] = getelementptr inbounds i32, ptr [[A]], i32 2
; CHECK-NEXT:   [[L:%.*]] = load i32, ptr [[GEP]], align 4
; CHECK-NEXT:   [[EXT:%.*]] = zext i32 [[L]] to i64
; CHECK-NEXT:   call void @use(
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

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
