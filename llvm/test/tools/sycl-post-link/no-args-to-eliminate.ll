; This test ensures that sycl-post-link doesn't crash when kernel parameter
; optimization info metadata is empty
;
; RUN: sycl-post-link -properties -emit-param-info -S < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop
;
; CHECK: [SYCL/kernel param opt]
; // Nothing should be here
; CHECK-EMPTY:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTSZ4mainEUlT_E0_() local_unnamed_addr #0  !kernel_arg_buffer_location !6 !spir_kernel_omit_args !6 {
entry:
  ret void
}

attributes #0 = { norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="d.cpp" "uniform-work-group-size"="true" }

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
