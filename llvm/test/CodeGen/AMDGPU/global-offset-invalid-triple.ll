; RUN: not --crash llc -march=amdgcn -mcpu=hawaii %s -o - 2>&1 | FileCheck %s
; ModuleID = 'global-offset-invalid-triple.bc'
; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.implicit.offset
source_filename = "global-offset-invalid-triple.ll"

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"

; This test checks that the pass does not run on nvcl triples.

declare i32 addrspace(5)* @llvm.amdgcn.implicit.offset()

define weak_odr dso_local i64 @_ZTS14other_function() {
  %1 = tail call i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
  %2 = getelementptr inbounds i32, i32 addrspace(5)* %1, i64 2
  %3 = load i32, i32 addrspace(5)* %2, align 4
  %4 = zext i32 %3 to i64
  ret i64 %4
}

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS14example_kernel() {
entry:
  %0 = call i64 @_ZTS14other_function()
  ret void
}

!amdgcn.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}

!0 = distinct !{void ()* @_ZTS14example_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
