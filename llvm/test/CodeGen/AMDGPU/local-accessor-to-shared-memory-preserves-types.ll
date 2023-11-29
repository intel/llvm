; RUN: opt -bugpoint-enable-legacy-pm -localaccessortosharedmemory %s -S -o - | FileCheck %s
; ModuleID = 'bitcasts.bc'
source_filename = "bitcasts.ll"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; This test checks that the transformation always bitcasts to the correct type.

; CHECK: @_ZTS14example_kernel_shared_mem = external addrspace(3) global [0 x i8], align 4

; Function Attrs: noinline
define amdgpu_kernel void @_ZTS14example_kernel(ptr addrspace(3) %a, ptr addrspace(3) %b, ptr addrspace(3) %c, ptr addrspace(3) %d) {
; CHECK: define amdgpu_kernel void @_ZTS14example_kernel(i32 %0, i32 %1, i32 %2, i32 %3) {
entry:
; CHECK: %4 = getelementptr inbounds [0 x i8], ptr addrspace(3) @_ZTS14example_kernel_shared_mem, i32 0, i32 %3
; CHECK: %5 = getelementptr inbounds [0 x i8], ptr addrspace(3) @_ZTS14example_kernel_shared_mem, i32 0, i32 %2
; CHECK: %6 = getelementptr inbounds [0 x i8], ptr addrspace(3) @_ZTS14example_kernel_shared_mem, i32 0, i32 %1
; CHECK: %7 = getelementptr inbounds [0 x i8], ptr addrspace(3) @_ZTS14example_kernel_shared_mem, i32 0, i32 %0
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

!amdgcn.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}

!0 = distinct !{ptr @_ZTS14example_kernel, !"kernel", i32 1}
; CHECK: !0 = distinct !{ptr @_ZTS14example_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
