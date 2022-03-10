; This test checks that the Local Accessor to Shared Memory pass does not run with the
; `nvptx64-nvidia-nvcl` triple.
; RUN: llc -march=nvptx64 -mcpu=sm_20 -sycl-enable-local-accessor < %s | FileCheck %s
; CHECK: .param .u64 .ptr .shared .align 1 _ZTS14example_kernel_param_0

; ModuleID = 'local-accessor-to-shared-memory-invalid-triple.ll'
source_filename = "local-accessor-to-shared-memory-invalid-triple.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-nvcl"

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS14example_kernel(i32 addrspace(3)* %a) {
entry:
  ret void
}

!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
!llvm.ident = !{!7, !8}
!nvvmir.version = !{!9}
!llvm.module.flags = !{!10, !11}

!0 = distinct !{void (i32 addrspace(3)*)* @_ZTS14example_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = !{i32 1, i32 2}
!6 = !{i32 4, i32 100000}
!7 = !{!"clang version 9.0.0"}
!8 = !{!"clang version 9.0.0"}
!9 = !{i32 1, i32 4}
!10 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 0]}
!11 = !{i32 1, !"wchar_size", i32 4}
