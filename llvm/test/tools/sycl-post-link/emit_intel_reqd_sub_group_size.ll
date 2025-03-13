; This test checks that the sycl-post-link tool correctly handles
; intel_reqd_sub_group_size metadata.

; RUN: sycl-post-link -properties -emit-program-metadata -device-globals -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files.table --check-prefixes CHECK-TABLE
; RUN: FileCheck %s -input-file=%t.files_0.prop --match-full-lines --check-prefixes CHECK-PROP

target triple = "amdgcn-amd-amdhsa"

!0 = !{i32 64}

define weak_odr amdgpu_kernel void @_ZTS7Kernel1(float %arg1) !intel_reqd_sub_group_size !0 {
  call void @foo(float %arg1)
  ret void
}

declare void @foo(float)

; CHECK-PROP: [SYCL/program metadata]
; CHECK-PROP-NEXT: _ZTS7Kernel1@reqd_sub_group_size=1|64

; CHECK-TABLE: [Code|Properties]
; CHECK-TABLE-NEXT: {{.*}}files_0.prop
; CHECK-TABLE-EMPTY:
