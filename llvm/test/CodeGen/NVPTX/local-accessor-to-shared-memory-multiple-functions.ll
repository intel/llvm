; RUN: opt -enable-new-pm=0 -localaccessortosharedmemory %s -S -o - | FileCheck %s
; ModuleID = 'multiple-functions.bc'
source_filename = "multiple-functions.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda-sycldevice"

; This test checks that the transformation does not break kernels which call other functions.

; CHECK: @_ZTS14example_kernel_shared_mem = external addrspace(3) global [0 x i8], align 4

define weak_odr dso_local void @_ZTS14other_function(i32 addrspace(3)* %a, i32 addrspace(1)* %b, i32 %c) {
; CHECK: define weak_odr dso_local void @_ZTS14other_function(i32 addrspace(3)* %a, i32 addrspace(1)* %b, i32 %c) {
  %1 = load i32, i32 addrspace(3)* %a
; CHECK: %1 = load i32, i32 addrspace(3)* %a
  %2 = load i32, i32 addrspace(1)* %b
; CHECK: %2 = load i32, i32 addrspace(1)* %b
  %3 = add i32 %c, %c
; CHECK: %3 = add i32 %c, %c
  ret void
}

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS14example_kernel(i32 addrspace(3)* %a, i32 addrspace(1)* %b, i32 %c) {
; CHECK: define weak_odr dso_local void @_ZTS14example_kernel(i32 %0, i32 addrspace(1)* %b, i32 %c) {
entry:
; CHECK: %1 = getelementptr inbounds [0 x i8], [0 x i8] addrspace(3)* @_ZTS14example_kernel_shared_mem, i32 0, i32 %0
; CHECK: %a = bitcast i8 addrspace(3)* %1 to i32 addrspace(3)*
   call void @_ZTS14other_function(i32 addrspace(3)* %a, i32 addrspace(1)* %b, i32 %c)
; CHECK: call void @_ZTS14other_function(i32 addrspace(3)* %a, i32 addrspace(1)* %b, i32 %c)
  ret void
}

!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
!nvvmir.version = !{!5}

!0 = distinct !{void (i32 addrspace(3)*, i32 addrspace(1)*, i32)* @_ZTS14example_kernel, !"kernel", i32 1}
; CHECK: !0 = distinct !{void (i32, i32 addrspace(1)*, i32)* @_ZTS14example_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = !{i32 1, i32 4}
