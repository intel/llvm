; This test checks that the Local Accessor to Shared Memory pass runs with the
; `nvptx64-nvidia-cuda` triple.
; RUN: llc -march=nvptx64 -mcpu=sm_20 -sycl-enable-local-accessor < %s | FileCheck --check-prefix=CHECK-OPT %s
; RUN: llc -march=nvptx64 -mcpu=sm_20 -sycl-enable-local-accessor=true < %s | FileCheck --check-prefix=CHECK-OPT %s
; RUN: llc -march=nvptx64 -mcpu=sm_20 < %s | FileCheck --check-prefix=CHECK-NO-OPT %s
; RUN: llc -march=nvptx64 -mcpu=sm_20 -sycl-enable-local-accessor=false < %s | FileCheck --check-prefix=CHECK-NO-OPT %s
; CHECK-OPT: .param .u32 _ZTS14example_kernel_param_0
; CHECK-NO-OPT-NOT: .param .u32 _ZTS14example_kernel_param_0

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
