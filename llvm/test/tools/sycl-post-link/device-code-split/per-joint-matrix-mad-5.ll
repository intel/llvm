; This test has 6 kernels:
; Kernel1 and Kernel2 have the same joint_matrix parameters
; Kernel3 and Kernel5 have the same joint_matrix and joint_matrix_mad parameters
; Kernel4 has the different joint_matrix parameters and same joint_matrix_mad parameters as Kernel3 and Kernel5
; Kernel6 has the same joint_matrix parameters as Kernel3 and Kernel5 and different joint_matrix_mad parameters

; The test is intended to check that sycl-post-link correctly separates kernels
; that use different sycl_joint_matrix_mad metadata

; RUN: sycl-post-link -properties -split=auto -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-IR-K3,CHECK-IR-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN: --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-IR-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefixes CHECK-IR-K1,CHECK-IR-K2 \
; RUN: --implicit-check-not Kernel3 --implicit-check-not Kernel4 \
; RUN:  --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_3.ll --check-prefixes CHECK-IR-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYMS-K3,CHECK-SYMS-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN: --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYMS-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefixes CHECK-SYMS-K1,CHECK-SYMS-K2 \
; RUN: --implicit-check-not Kernel3 --implicit-check-not Kernel4 \
; RUN:  --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_3.sym --check-prefixes CHECK-SYMS-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6

; RUN: sycl-module-split -split=auto -S %s -o %t2
; RUN: FileCheck %s -input-file=%t2_0.ll --check-prefixes CHECK-IR-K3,CHECK-IR-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN: --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_1.ll --check-prefixes CHECK-IR-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t2_2.ll --check-prefixes CHECK-IR-K1,CHECK-IR-K2 \
; RUN: --implicit-check-not Kernel3 --implicit-check-not Kernel4 \
; RUN:  --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_3.ll --check-prefixes CHECK-IR-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_0.sym --check-prefixes CHECK-SYMS-K3,CHECK-SYMS-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN: --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_1.sym --check-prefixes CHECK-SYMS-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t2_2.sym --check-prefixes CHECK-SYMS-K1,CHECK-SYMS-K2 \
; RUN: --implicit-check-not Kernel3 --implicit-check-not Kernel4 \
; RUN:  --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_3.sym --check-prefixes CHECK-SYMS-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6

; RUN: sycl-post-link -properties -split=source -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-IR-K3,CHECK-IR-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN: --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-IR-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefixes CHECK-IR-K1,CHECK-IR-K2 \
; RUN: --implicit-check-not Kernel3 --implicit-check-not Kernel4 \
; RUN:  --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_3.ll --check-prefixes CHECK-IR-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYMS-K3,CHECK-SYMS-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN: --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYMS-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefixes CHECK-SYMS-K1,CHECK-SYMS-K2 \
; RUN: --implicit-check-not Kernel3 --implicit-check-not Kernel4 \
; RUN:  --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_3.sym --check-prefixes CHECK-SYMS-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6

; RUN: sycl-module-split -split=source -S %s -o %t2
; RUN: FileCheck %s -input-file=%t2_0.ll --check-prefixes CHECK-IR-K3,CHECK-IR-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN: --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_1.ll --check-prefixes CHECK-IR-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t2_2.ll --check-prefixes CHECK-IR-K1,CHECK-IR-K2 \
; RUN: --implicit-check-not Kernel3 --implicit-check-not Kernel4 \
; RUN:  --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_3.ll --check-prefixes CHECK-IR-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_0.sym --check-prefixes CHECK-SYMS-K3,CHECK-SYMS-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN: --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_1.sym --check-prefixes CHECK-SYMS-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t2_2.sym --check-prefixes CHECK-SYMS-K1,CHECK-SYMS-K2 \
; RUN: --implicit-check-not Kernel3 --implicit-check-not Kernel4 \
; RUN:  --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_3.sym --check-prefixes CHECK-SYMS-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6

; RUN: sycl-post-link -properties -split=kernel -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-IR-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-IR-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefixes CHECK-IR-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_3.ll --check-prefixes CHECK-IR-K3 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_4.ll --check-prefixes CHECK-IR-K2 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel3 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_5.ll --check-prefixes CHECK-IR-K1 \
; RUN: --implicit-check-not Kernel12 --implicit-check-not Kernel3 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYMS-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYMS-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefixes CHECK-SYMS-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_3.sym --check-prefixes CHECK-SYMS-K3 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_4.sym --check-prefixes CHECK-SYMS-K2 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel3 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t_5.sym --check-prefixes CHECK-SYMS-K1 \
; RUN: --implicit-check-not Kernel12 --implicit-check-not Kernel3 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6

; RUN: sycl-module-split -split=kernel -S %s -o %t2
; RUN: FileCheck %s -input-file=%t2_0.ll --check-prefixes CHECK-IR-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t2_1.ll --check-prefixes CHECK-IR-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_2.ll --check-prefixes CHECK-IR-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_3.ll --check-prefixes CHECK-IR-K3 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_4.ll --check-prefixes CHECK-IR-K2 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel3 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_5.ll --check-prefixes CHECK-IR-K1 \
; RUN: --implicit-check-not Kernel12 --implicit-check-not Kernel3 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_0.sym --check-prefixes CHECK-SYMS-K6 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel5
; RUN: FileCheck %s -input-file=%t2_1.sym --check-prefixes CHECK-SYMS-K5 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel4 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_2.sym --check-prefixes CHECK-SYMS-K4 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel3 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_3.sym --check-prefixes CHECK-SYMS-K3 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel2 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_4.sym --check-prefixes CHECK-SYMS-K2 \
; RUN: --implicit-check-not Kernel1 --implicit-check-not Kernel3 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6
; RUN: FileCheck %s -input-file=%t2_5.sym --check-prefixes CHECK-SYMS-K1 \
; RUN: --implicit-check-not Kernel12 --implicit-check-not Kernel3 \
; RUN:  --implicit-check-not Kernel4 --implicit-check-not Kernel5 --implicit-check-not Kernel6

; CHECK-IR-K1: define {{.*}} @Kernel1
; CHECK-IR-K2: define {{.*}} @Kernel2
; CHECK-IR-K3: define {{.*}} @Kernel3
; CHECK-IR-K4: define {{.*}} @Kernel4
; CHECK-IR-K5: define {{.*}} @Kernel5
; CHECK-IR-K6: define {{.*}} @Kernel6
; CHECK-SYMS-K1: Kernel1
; CHECK-SYMS-K2: Kernel2
; CHECK-SYMS-K3: Kernel3
; CHECK-SYMS-K4: Kernel4
; CHECK-SYMS-K5: Kernel5
; CHECK-SYMS-K6: Kernel6

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$Kernel1 = comdat any

$Kernel2 = comdat any

$Kernel3 = comdat any

$Kernel4 = comdat any

$Kernel5 = comdat any

$Kernel6 = comdat any

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @Kernel1() local_unnamed_addr #0 comdat !srcloc !5 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_joint_matrix !11 !sycl_kernel_omit_args !6 {
entry:
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @Kernel2() local_unnamed_addr #0 comdat !srcloc !8 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_joint_matrix !11 !sycl_kernel_omit_args !6 {
entry:
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @Kernel3() local_unnamed_addr #0 comdat !srcloc !9 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_joint_matrix !11 !sycl_joint_matrix_mad !10 !sycl_kernel_omit_args !6 {
entry:
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @Kernel4() local_unnamed_addr #0 comdat !srcloc !9 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_joint_matrix !12 !sycl_joint_matrix_mad !10 !sycl_kernel_omit_args !6 {
entry:
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @Kernel5() local_unnamed_addr #0 comdat !srcloc !9 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_joint_matrix !11 !sycl_joint_matrix_mad !10 !sycl_kernel_omit_args !6 {
entry:
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @Kernel6() local_unnamed_addr #0 comdat !srcloc !9 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_joint_matrix !11 !sycl_joint_matrix_mad !7 !sycl_kernel_omit_args !6 {
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
!11 = !{!"matrix_type::fp32,use::a,12,32;matrix_type::fp64,use::b,67,21"}
!12 = !{!"matrix_type::fp32,use::a,12,32"}
