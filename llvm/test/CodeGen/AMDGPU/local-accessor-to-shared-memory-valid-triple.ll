; This test checks that the Local Accessor to Shared Memory pass runs with the
; `amdgcn-amd-amdhsa` triple and does not if the option is not present.
; RUN: llc -mtriple=amdgcn-amd-amdhsa -sycl-enable-local-accessor < %s | FileCheck --check-prefix=CHECK-OPT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -sycl-enable-local-accessor=true < %s | FileCheck --check-prefix=CHECK-OPT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s | FileCheck --check-prefix=CHECK-NO-OPT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -sycl-enable-local-accessor=false < %s | FileCheck --check-prefix=CHECK-NO-OPT %s

; ModuleID = 'local-accessor-to-shared-memory-valid-triple.ll'
source_filename = "local-accessor-to-shared-memory-valid-triple.ll"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; CHECK-OPT: .globl	_ZTS14example_kernel
; CHECK-OPT: - .args:
; CHECK-OPT-NOT: .address_space: local
; CHECK-OPT-NEXT: .offset: 0
; CHECK-OPT-NEXT: .size: 4
; CHECK-OPT-NEXT: .value_kind:     by_value
; CHECK-NO-OPT: .globl	_ZTS14example_kernel
; CHECK-NO-OPT: - .args:
; CHECK-NO-OPT-NEXT: .address_space: local
; CHECK-NO-OPT-NEXT: .name: a
; CHECK-NO-OPT-NEXT: .offset: 0
; CHECK-NO-OPT-NEXT: .pointee_align: 1
; CHECK-NO-OPT-NEXT: .size: 4
; CHECK-NO-OPT-NEXT: .value_kind:     dynamic_shared_pointer
; Function Attrs: noinline
define amdgpu_kernel void @_ZTS14example_kernel(i32 addrspace(3)* %a) {
entry:
  %0 = load i32, i32 addrspace(3)* %a
  ret void
}
