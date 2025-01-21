; This test ensures that sycl-post-link will optimize away
; unused functions that are safe to remove even if there are no
; splits.
; RUN: sycl-post-link -properties -split-esimd -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.ll --implicit-check-not=foo

; CHECK: target datalayout
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define linkonce_odr dso_local spir_func void @foo() local_unnamed_addr #0 {
entry:
  ret void
}

; CHECK: _ZTSZ4mainEUlT_E0_
; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ4mainEUlT_E0_() local_unnamed_addr #0  !kernel_arg_buffer_location !6 !spir_kernel_omit_args !6 {
entry:
  ret void
}

; CHECK: attributes #0
attributes #0 = { norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="file.cpp" "uniform-work-group-size"="true" }

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2, !3}
!llvm.module.flags = !{!4, !5}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 14.0.0"}
!3 = !{!"clang version 14.0.0"}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"frame-pointer", i32 2}
!6 = !{}
