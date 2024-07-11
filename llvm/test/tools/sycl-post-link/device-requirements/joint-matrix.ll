; Original code:
;
; // Compiled with clang++ -fsycl -fsycl-device-only -fno-sycl-instrument-device-code -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4 -O2 -S -emit-llvm -o - %s
; int main() {
;   sycl::queue q;
;   q.submit([&](sycl::handler &cgh) {
;     cgh.parallel_for(nd_range<2>({1, 16}, {1, 16}), [=](nd_item<2> it) {
;       joint_matrix<sycl::sub_group, float, use::a, 8, 16, layout::row_major> a;
;       joint_matrix<sycl::sub_group, float, use::b, 16, 16, layout::row_major> b;
;       joint_matrix<sycl::sub_group, float, use::accumulator, 8, 16> c;
;       sub_group sg = it.get_sub_group();
;       joint_matrix_mad(sg, c, a, b, c);
;     });
;   });
;   q.submit([&](sycl::handler &cgh) {
;     cgh.parallel_for(nd_range<2>({1, 16}, {1, 16}), [=](nd_item<2> it) {
;       joint_matrix<sycl::sub_group, double, use::a, 16, 16, layout::row_major>
;           a;
;     });
;   });
;   return 0;
; }

; RUN: sycl-post-link -properties -split=kernel %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop --check-prefix CHECK-PROP-KERNEL-SPLIT-0
; RUN: FileCheck %s -input-file=%t_1.prop --check-prefix CHECK-PROP-KERNEL-SPLIT-1

; CHECK-PROP-KERNEL-SPLIT-0: [SYCL/device requirements]
; CHECK-PROP-KERNEL-SPLIT-0: joint_matrix=2|gMAAAAAAAAQbhRncph3X0lHclpjOmB3MywSdzVmO6EGL4wSM2sTbhRncph3X0lHclpjOmB3MywSdzVmO6E2YjVXb1xWY09mcsgDLxYzOtFGdylGefRXewVmO6YGczIDL1NXZ6ojYsEjNsEjN
; CHECK-PROP-KERNEL-SPLIT-0-NEXT: joint_matrix_mad=2|4JAAAAAAAAQbhRncph3X0lHclpjOmB3MywSbhRncph3X0lHclpjOmB3MywSbhRncph3X0lHclpjOmB3MywSbhRncph3X0lHclpjOmB3MywCOsEjNsEjN

; CHECK-PROP-KERNEL-SPLIT-1: [SYCL/device requirements]
; CHECK-PROP-KERNEL-SPLIT-1: joint_matrix=2|wDAAAAAAAAQbhRncph3X0lHclpjOmBnN0wSdzVmO6EGLxYDLxYD

; ModuleID = '/tmp/test.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi2EEEE_ = comdat any

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_EUlNS0_7nd_itemILi2EEEE_ = comdat any

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_7nd_itemILi2EEEE_() local_unnamed_addr #0 comdat !srcloc !5 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_joint_matrix !7 !sycl_joint_matrix_mad !8 !sycl_kernel_omit_args !6 {
entry:
  %call.i.i = tail call spir_func noundef target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z27__spirv_JointMatrixMadINTELIffLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS1_1ELS1_2ELNS0_12MatrixLayoutE0ELS2_0ELS2_3ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT9_EXT10_EXT6_EEEPNS5_IT_XT1_EXT2_EXT7_EXT10_EXT4_EEEPNS5_IS9_XT2_EXT3_EXT8_EXT10_EXT5_EEES8_S4_(target("spirv.JointMatrixINTEL", float, 8, 16, 0, 3, 0) noundef undef, target("spirv.JointMatrixINTEL", float, 16, 16, 0, 3, 1) noundef undef, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) noundef undef, i32 noundef 3) #3
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z27__spirv_JointMatrixMadINTELIffLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS1_1ELS1_2ELNS0_12MatrixLayoutE0ELS2_0ELS2_3ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT9_EXT10_EXT6_EEEPNS5_IT_XT1_EXT2_EXT7_EXT10_EXT4_EEEPNS5_IS9_XT2_EXT3_EXT8_EXT10_EXT5_EEES8_S4_(target("spirv.JointMatrixINTEL", float, 8, 16, 0, 3, 0) noundef, target("spirv.JointMatrixINTEL", float, 16, 16, 0, 3, 1) noundef, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_EUlNS0_7nd_itemILi2EEEE_() local_unnamed_addr #2 comdat !srcloc !9 !kernel_arg_buffer_location !6 !sycl_fixed_targets !6 !sycl_joint_matrix !10 !sycl_kernel_omit_args !6 {
entry:
  ret void
}

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="sss.cpp" "sycl-optlevel"="2" "uniform-work-group-size"="true" }
attributes #1 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="sss.cpp" "sycl-optlevel"="2" "uniform-work-group-size"="true" }
attributes #3 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!""}
!5 = !{i32 1091}
!6 = !{}
!7 = !{!"matrix_type::fp32,use::a,8,16;matrix_type::fp32,use::accumulator,8,16;matrix_type::fp32,use::b,16,16"}
!8 = !{!"matrix_type::fp32,matrix_type::fp32,matrix_type::fp32,matrix_type::fp32,8,16,16"}
!9 = !{i32 1529}
!10 = !{!"matrix_type::fp64,use::a,16,16"}
