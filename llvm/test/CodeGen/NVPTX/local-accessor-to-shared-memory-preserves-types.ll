; RUN: opt -bugpoint-enable-legacy-pm -localaccessortosharedmemory %s -S -o - | FileCheck %s
; ModuleID = 'bitcasts.bc'
source_filename = "bitcasts.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; This test checks that the transformation always bitcasts to the correct type.

; CHECK: @_ZTS14example_kernel_shared_mem = external addrspace(3) global [0 x i8], align 4

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS14example_kernel(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d) {
; CHECK: define weak_odr dso_local void @_ZTS14example_kernel(i32 %0, i32 %1, i32 %2, i32 %3) {
entry:
; CHECK: %4 = getelementptr inbounds [0 x i8], ptr addrspace(3) @_ZTS14example_kernel_shared_mem, i32 0, i32 %3
; CHECK: %d = bitcast ptr addrspace(3) %4 to ptr addrspace(3)
; CHECK: %5 = getelementptr inbounds [0 x i8], ptr addrspace(3) @_ZTS14example_kernel_shared_mem, i32 0, i32 %2
; CHECK: %c = bitcast ptr addrspace(3) %5 to ptr addrspace(3)
; CHECK: %6 = getelementptr inbounds [0 x i8], ptr addrspace(3) @_ZTS14example_kernel_shared_mem, i32 0, i32 %1
; CHECK: %b = bitcast ptr addrspace(3) %6 to ptr addrspace(3)
; CHECK: %7 = getelementptr inbounds [0 x i8], ptr addrspace(3) @_ZTS14example_kernel_shared_mem, i32 0, i32 %0
; CHECK: %a = bitcast ptr addrspace(3) %7 to ptr addrspace(3)
  %0 = load i32, ptr addrspace(3) %a
; CHECK: %8 = load i32, ptr addrspace(3) %a
  %1 = load i64, ptr addrspace(3) %b
; CHECK: %9 = load i64, ptr addrspace(3) %b
  %2 = load i16, ptr addrspace(3) %c
; CHECK: %10 = load i16, ptr addrspace(3) %c
  %3 = load i8, ptr addrspace(3) %d
; CHECK: %11 = load i8, ptr addrspace(3) %d
  ret void
}

!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
!nvvmir.version = !{!5}

!0 = distinct !{ptr @_ZTS14example_kernel, !"kernel", i32 1}
; CHECK: !0 = distinct !{ptr @_ZTS14example_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = !{i32 1, i32 4}
