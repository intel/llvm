; RUN: opt -enable-new-pm=0 -globaloffset %s -S -o - | FileCheck %s
; ModuleID = 'global-offset-simple.bc'
source_filename = "global-offset-simple.ll"

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; This test checks that the transformation is applied in the basic case.

declare i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
; CHECK-NOT: llvm.amdgcn.implicit.offset

define weak_odr dso_local i64 @_ZTS14other_function() {
; CHECK: define weak_odr dso_local i64 @_ZTS14other_function(i32 addrspace(5)* %0) {
; CHECK: %2 = getelementptr inbounds i32, i32 addrspace(5)* %0, i64 2
  %1 = tail call i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
; CHECK-NOT: tail call i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
  %2 = getelementptr inbounds i32, i32 addrspace(5)* %1, i64 2
  %3 = load i32, i32 addrspace(5)* %2, align 4
  %4 = zext i32 %3 to i64
  ret i64 %4
}

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS14example_kernel() {
entry:
; CHECK:   %0 = alloca [3 x i32], align 4, addrspace(5)
; CHECK:   %1 = bitcast [3 x i32] addrspace(5)* %0 to i8 addrspace(5)*
; CHECK:   call void @llvm.memset.p5i8.i64(i8 addrspace(5)* nonnull align 4 dereferenceable(12) %1, i8 0, i64 12, i1 false)
; CHECK:   %2 = getelementptr inbounds [3 x i32], [3 x i32] addrspace(5)* %0, i32 0, i32 0
  %0 = call i64 @_ZTS14other_function()
; CHECK: %3 = call i64 @_ZTS14other_function(i32 addrspace(5)* %2)
  ret void
}

; CHECK: define weak_odr dso_local void @_ZTS14example_kernel_with_offset([3 x i32]* byref([3 x i32]) %0) {
; CHECK: entry:
; CHECK:   %1 = alloca [3 x i32], align 4, addrspace(5)
; CHECK:   %2 = bitcast [3 x i32] addrspace(5)* %1 to i32 addrspace(5)*
; CHECK:   call void @llvm.memcpy.p5i8.p4i8.i64(i8 addrspace(5)* align 4 %4, i8 addrspace(4)* align 1 %3, i64 12, i1 false)
; CHECK:   %5 = call i64 @_ZTS14other_function(i32 addrspace(5)* %2)
; CHECK:   ret void
; CHECK: }

!amdgcn.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
; CHECK: !amdgcn.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3, !5}

!0 = distinct !{void ()* @_ZTS14example_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
; CHECK: !5 = !{void ([3 x i32]*)* @_ZTS14example_kernel_with_offset, !"kernel", i32 1}
