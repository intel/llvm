; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This test checks that the transformation is applied to kernels found 

declare ptr @llvm.nvvm.implicit.offset()
; CHECK-NOT: llvm.nvvm.implicit.offset

define i64 @_ZTS14other_function() {
  %1 = tail call ptr @llvm.nvvm.implicit.offset()
  %2 = getelementptr inbounds i32, ptr %1, i64 2
  %3 = load i32, ptr %2, align 4
  %4 = zext i32 %3 to i64
  ret i64 %4
}

define ptx_kernel void @_ZTS14example_kernel() {
entry:
  %0 = call i64 @_ZTS14other_function()
  ret void
}

!llvm.module.flags = !{!0}
!nvvm.annotations = !{!1, !2, !2, !1}

!0 = !{i32 1, !"sycl-device", i32 1}
!1 = distinct !{ptr @_ZTS14example_kernel, !"maxnreg", i32 256, !"kernel", i32 1}
!2 = !{ptr @_ZTS14example_kernel, !"maxntidx", i32 8, !"maxntidy", i32 16, !"maxntidz", i32 32}
