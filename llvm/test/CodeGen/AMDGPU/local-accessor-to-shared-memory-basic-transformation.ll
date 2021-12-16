; RUN: opt -enable-new-pm=0 -localaccessortosharedmemory -sycl-enable-local-accessor %s -S -o - | FileCheck %s
; ModuleID = 'basic-transformation.bc'
source_filename = "basic-transformation.ll"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; This test checks that the transformation is applied in the basic case.

; CHECK: @_ZTS14example_kernel_shared_mem = external addrspace(3) global [0 x i8], align 4

; Function Attrs: noinline
define amdgpu_kernel void @_ZTS14example_kernel(i32 addrspace(3)* %a, i32 addrspace(1)* %b, i32 %c) {
; CHECK: define amdgpu_kernel void @_ZTS14example_kernel(i32 %0, i32 addrspace(1)* %b, i32 %c) {
entry:
; CHECK: %1 = getelementptr inbounds [0 x i8], [0 x i8] addrspace(3)* @_ZTS14example_kernel_shared_mem, i32 0, i32 %0
; CHECK: %a = bitcast i8 addrspace(3)* %1 to i32 addrspace(3)*
  %0 = load i32, i32 addrspace(3)* %a
; CHECK: %2 = load i32, i32 addrspace(3)* %a
  %1 = load i32, i32 addrspace(1)* %b
; CHECK: %3 = load i32, i32 addrspace(1)* %b
  %2 = add i32 %c, %c
; CHECK: %4 = add i32 %c, %c
  ret void
}
