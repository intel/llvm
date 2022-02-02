; RUN: opt -enable-new-pm=0 -localaccessortosharedmemory -sycl-enable-local-accessor %s -S -o - | FileCheck %s
; ModuleID = 'no-entry-points.bc'
source_filename = "no-entry-points.ll"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; This test checks that no transformation is applied when there are no entry points.

; Function Attrs: noinline
define void @_ZTS14example_kernel(i32 addrspace(3)* %a, i32 addrspace(1)* %b, i32 %c) {
; CHECK: define void @_ZTS14example_kernel(i32 addrspace(3)* %a, i32 addrspace(1)* %b, i32 %c) {
entry:
  %0 = load i32, i32 addrspace(3)* %a
; CHECK: %0 = load i32, i32 addrspace(3)* %a
  %1 = load i32, i32 addrspace(1)* %b
; CHECK: %1 = load i32, i32 addrspace(1)* %b
  %2 = add i32 %c, %c
; CHECK: %2 = add i32 %c, %c
  ret void
}
