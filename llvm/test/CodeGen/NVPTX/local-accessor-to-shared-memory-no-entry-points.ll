; RUN: opt -enable-new-pm=0 -localaccessortosharedmemory -sycl-enable-local-accessor %s -S -o - | FileCheck %s
; ModuleID = 'no-entry-points.bc'
source_filename = "no-entry-points.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This test checks that no transformation is applied when there are no entry points.

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS14example_kernel(i32 addrspace(3)* %a, i32 addrspace(1)* %b, i32 %c) {
; CHECK: define weak_odr dso_local void @_ZTS14example_kernel(i32 addrspace(3)* %a, i32 addrspace(1)* %b, i32 %c) {
entry:
  %0 = load i32, i32 addrspace(3)* %a
; CHECK: %0 = load i32, i32 addrspace(3)* %a
  %1 = load i32, i32 addrspace(1)* %b
; CHECK: %1 = load i32, i32 addrspace(1)* %b
  %2 = add i32 %c, %c
; CHECK: %2 = add i32 %c, %c
  ret void
}

!nvvm.annotations = !{!0, !1, !0, !2, !2, !2, !2, !3, !3, !2}
!nvvmir.version = !{!4}

!0 = !{null, !"align", i32 8}
!1 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!2 = !{null, !"align", i32 16}
!3 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!4 = !{i32 1, i32 4}
