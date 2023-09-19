; This test checks that the Local Accessor to Shared Memory pass runs with the
; `amdgcn-amd-amdhsa` triple and does not if the option is not present.
; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s | FileCheck --check-prefix=CHECK-OPT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s | FileCheck --check-prefix=CHECK-OPT %s

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
; Function Attrs: noinline
define amdgpu_kernel void @_ZTS14example_kernel(ptr addrspace(3) %a) {
entry:
  %0 = load i32, ptr addrspace(3) %a
  ret void
}

!amdgcn.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
!llvm.ident = !{!7, !8}
!llvm.module.flags = !{!9, !10}

!0 = distinct !{ptr @_ZTS14example_kernel, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = !{i32 1, i32 2}
!6 = !{i32 4, i32 100000}
!7 = !{!"clang version 9.0.0"}
!8 = !{!"clang version 9.0.0"}
!9 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 0]}
!10 = !{i32 1, !"wchar_size", i32 4}
