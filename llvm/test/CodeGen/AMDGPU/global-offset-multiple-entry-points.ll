; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s
; ModuleID = 'global-offset-multiple-entry-points.bc'
source_filename = "global-offset-multiple-entry-points.ll"

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; This test checks that the pass works with multiple entry points.

declare ptr addrspace(5) @llvm.amdgcn.implicit.offset()
; CHECK-NOT: declare ptr addrspace(5) @llvm.amdgcn.implicit.offset()

; This function is a kernel entry point that does not use global offset. It will
; not get a clone with a global offset parameter.
; Function Attrs: noinline
define weak_odr dso_local void @_ZTS12third_kernel() {
entry:
  ret void
}

define weak_odr dso_local i64 @_ZTS15common_function() {
; CHECK: define weak_odr dso_local i64 @_ZTS15common_function() {
  %1 = tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
; CHECK-NOT: tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %2 = getelementptr inbounds i32, ptr addrspace(5) %1, i64 2
  %3 = load i32, ptr addrspace(5) %2, align 4
  %4 = zext i32 %3 to i64
; CHECK: %1 = zext i32 0 to i64
  ret i64 %4
}

; CHECK: define weak_odr dso_local i64 @_ZTS15common_function_with_offset(ptr addrspace(5) %0) {  
; CHECK: %2 = getelementptr inbounds i32, ptr addrspace(5) %0, i64 2
; CHECK: %3 = load i32, ptr addrspace(5) %2, align 4
; CHECK: %4 = zext i32 %3 to i64
; CHECK: ret i64 %4
; CHECK: }

define weak_odr dso_local i64 @_ZTS14first_function() {
; CHECK: define weak_odr dso_local i64 @_ZTS14first_function() {
  %1 = call i64 @_ZTS15common_function()
; CHECK: %1 = call i64 @_ZTS15common_function()
  ret i64 %1
}

; CHECK: define weak_odr dso_local i64 @_ZTS14first_function_with_offset(ptr addrspace(5) %0) {
; CHECK: %2 = call i64 @_ZTS15common_function_with_offset(ptr addrspace(5) %0)
; CHECK: ret i64 %2

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS12first_kernel() {
entry:
  %0 = call i64 @_ZTS14first_function()
; CHECK: %0 = call i64 @_ZTS14first_function()
  ret void
}

; CHECK: define weak_odr dso_local void @_ZTS12first_kernel_with_offset(ptr byref([3 x i32]) %0) {
; CHECK: entry:
; CHECK:   %1 = alloca [3 x i32], align 4, addrspace(5)
; CHECK:   %2 = addrspacecast ptr %0 to ptr addrspace(4)
; CHECK:   call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 4 %1, ptr addrspace(4) align 1 %2, i64 12, i1 false)
; CHECK:   %3 = call i64 @_ZTS14first_function_with_offset(ptr addrspace(5) %1)
; CHECK:   ret void
; CHECK: }

define weak_odr dso_local i64 @_ZTS15second_function() {
; CHECK: define weak_odr dso_local i64 @_ZTS15second_function() {
  %1 = call i64 @_ZTS15common_function()
; CHECK: %1 = call i64 @_ZTS15common_function()
  ret i64 %1
}

; CHECK: define weak_odr dso_local i64 @_ZTS15second_function_with_offset(ptr addrspace(5) %0) { 
; CHECK: %2 = call i64 @_ZTS15common_function_with_offset(ptr addrspace(5) %0)
; CHECK: ret i64 %2

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS13second_kernel() {
entry:
  %0 = call i64 @_ZTS15second_function()
; CHECK: %0 = call i64 @_ZTS15second_function()
  ret void
}

; CHECK: define weak_odr dso_local void @_ZTS13second_kernel_with_offset(ptr byref([3 x i32]) %0) {
; CHECK: entry:
; CHECK:   %1 = alloca [3 x i32], align 4, addrspace(5)
; CHECK:   %2 = addrspacecast ptr %0 to ptr addrspace(4)
; CHECK:   call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 4 %1, ptr addrspace(4) align 1 %2, i64 12, i1 false)
; CHECK:   %3 = call i64 @_ZTS15second_function_with_offset(ptr addrspace(5) %1)
; CHECK:   ret void
; CHECK: }

; This function doesn't get called by a kernel entry point.
define weak_odr dso_local i64 @_ZTS15no_entry_point() {
; CHECK: define weak_odr dso_local i64 @_ZTS15no_entry_point() {
  %1 = tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
; CHECK-NOT: tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %2 = getelementptr inbounds i32, ptr addrspace(5) %1, i64 2
  %3 = load i32, ptr addrspace(5) %2, align 4
  %4 = zext i32 %3 to i64
; CHECK: %1 = zext i32 0 to i64
  ret i64 %4
}

; CHECK: define weak_odr dso_local i64 @_ZTS15no_entry_point_with_offset(ptr addrspace(5) %0) {
; CHECK: %2 = getelementptr inbounds i32, ptr addrspace(5)  %0, i64 2
; CHECK: %3 = load i32, ptr addrspace(5) %2, align 4
; CHECK: %4 = zext i32 %3 to i64
; CHECK: ret i64 %4
; CHECK: }

!amdgcn.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3, !5, !6}
; CHECK: !amdgcn.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3, !5, !6, !7, !8}

!0 = distinct !{ptr @_ZTS12first_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = distinct !{ptr @_ZTS13second_kernel, !"kernel", i32 1}
!6 = distinct !{ptr @_ZTS12third_kernel, !"kernel", i32 1}
; CHECK: !7 = !{ptr @_ZTS13second_kernel_with_offset, !"kernel", i32 1}
; CHECK: !8 = !{ptr @_ZTS12first_kernel_with_offset, !"kernel", i32 1}
