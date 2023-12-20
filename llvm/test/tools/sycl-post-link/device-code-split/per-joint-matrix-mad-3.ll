; This test is intended to check that we do not perform per-joint-matrix-mad
; split if it was disabled through one or another sycl-post-link option

; RUN: sycl-post-link -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table --check-prefix CHECK-TABLE
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefix CHECK-IR
;
; -lower-esimd is needed so sycl-post-link does not complain about no actions
; specified
; RUN: sycl-post-link -lower-esimd -ir-output-only -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll --check-prefix CHECK-IR

; We expect to see only one module generated:
;
; CHECK-TABLE: Code
; CHECK-TABLE-NEXT: _0.ll
; CHECK-TABLE-EMPTY:

; CHECK-IR-DAG: define weak_odr dso_local spir_kernel void @Kernel1
; CHECK-IR-DAG: define weak_odr dso_local spir_kernel void @Kernel2
; CHECK-IR-DAG: define weak_odr dso_local spir_kernel void @Kernel3

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$Kernel1 = comdat any

$Kernel2 = comdat any

$Kernel3 = comdat any

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @Kernel1() local_unnamed_addr #0 comdat !srcloc !5 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_joint_matrix !7 !sycl_kernel_omit_args !6 {
entry:
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @Kernel2() local_unnamed_addr #0 comdat !srcloc !8 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_joint_matrix !7 !sycl_kernel_omit_args !6 {
entry:
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @Kernel3() local_unnamed_addr #0 comdat !srcloc !9 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_joint_matrix !10 !sycl_kernel_omit_args !6 {
entry:
  ret void
}

attributes #0 = { mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test.cpp" "sycl-optlevel"="2" "sycl-single-task" "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!""}
!5 = !{i32 1037}
!6 = !{}
!7 = !{!"matrix_type::sint8,matrix_type::sint8,matrix_type::sint32,matrix_type::sint32,12,48,12"}
!8 = !{i32 1301}
!9 = !{i32 1859}
!10 = !{!"matrix_type::sint8,matrix_type::sint8,matrix_type::sint32,matrix_type::sint32,12,48,28"}
