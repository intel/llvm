; This test checks that the Local Accessor to Shared Memory pass runs with the
; `amdgcn-amd-amdhsa` triple, but not with `amdgcn-amd-amdpal`.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -sycl-enable-local-accessor < %s | FileCheck --check-prefix=CHECK-VALID %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -sycl-enable-local-accessor < %s | FileCheck --check-prefix=CHECK-INVALID %s

; ModuleID = 'local-accessor-to-shared-memory-triple.ll'
source_filename = "local-accessor-to-shared-memory-triple.ll"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; CHECK-VALID: .globl	_ZTS14example_kernel
; CHECK-VALID: - .args:
; CHECK-VALID-NOT: .address_space: local
; CHECK-VALID-NEXT: .offset: 0
; CHECK-VALID-NEXT: .size: 4

; CHECK-INVALID: amdpal.pipelines:
; CHECK-INVALID-NOT: - .args:

; Function Attrs: noinline
define amdgpu_kernel void @_ZTS14example_kernel(i32 addrspace(3)* %a) {
entry:
  %0 = load i32, i32 addrspace(3)* %a
  ret void
}
