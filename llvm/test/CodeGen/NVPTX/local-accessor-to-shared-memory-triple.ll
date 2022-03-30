; This test checks that the Local Accessor to Shared Memory pass runs with the
; `nvptx64-nvidia-cuda` triple.
; RUN: llc -mtriple=nvptx64-nvidia-cuda -sycl-enable-local-accessor < %s | FileCheck --check-prefix=CHECK-VALID %s
; RUN: llc -mtriple=nvptx64-nvidia-nvcl -sycl-enable-local-accessor < %s | FileCheck --check-prefix=CHECK-INVALID %s
; CHECK-VALID: .param .u32 _ZTS14example_kernel_param_0
; CHECK-INVALID: .param .u64 .ptr .shared .align 1 _ZTS14example_kernel_param_0

; ModuleID = 'local-accessor-to-shared-memory-valid-triple.ll'
source_filename = "local-accessor-to-shared-memory-valid-triple.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS14example_kernel(i32 addrspace(3)* %a) {
entry:
  %0 = load i32, i32 addrspace(3)* %a
  ret void
}

!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
!nvvmir.version = !{!5}

!0 = distinct !{void (i32 addrspace(3)*)* @_ZTS14example_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = !{i32 1, i32 4}
