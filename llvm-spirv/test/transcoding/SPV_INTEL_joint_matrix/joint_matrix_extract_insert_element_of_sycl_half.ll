; RUN: llvm-as %s -o %t.bc

; RUN: llvm-spirv -s %t.bc -o %t.regularized.bc
; RUN: llvm-dis %t.regularized.bc -o %t.regularized.ll
; RUN: FileCheck < %t.regularized.ll %s --check-prefix=CHECK-REGULARIZED

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_joint_matrix -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-REGULARIZED: %[[#ExtractElementCall:]] = call spir_func half @_Z28__spirv_VectorExtractDynamicIN2cl4sycl6detail9half_impl4halfELm8ELm16ELN5__spv12MatrixLayoutE0ELNS5_5Scope4FlagE3EET_PNS5_24__spirv_JointMatrixINTELIS9_XT0_EXT1_EXT2_EXT3_EEEm(%spirv.JointMatrixINTEL._half_8_16_0_3 addrspace(4)* align 2{{.*}}, i64{{.*}})
; CHECK-REGULARIZED: %[[#GEP:]] = getelementptr inbounds %"class.cl::sycl::detail::half_impl::half", %"class.cl::sycl::detail::half_impl::half" addrspace(4)*{{.*}}, i32 0, i32 0
; CHECK-REGULARIZED: store half %[[#ExtractElementCall]], half addrspace(4)* %[[#GEP]]
; CHECK-REGULARIZED: %[[#GEP:]] = getelementptr inbounds %"class.cl::sycl::detail::half_impl::half", %"class.cl::sycl::detail::half_impl::half"*{{.*}}, i32 0, i32 0
; CHECK-REGULARIZED: %[[#Component:]] = load half, half*{{.*}}, align 2
; CHECK-REGULARIZED: call spir_func %spirv.JointMatrixINTEL._half_8_16_0_3 addrspace(4)* @_Z27__spirv_VectorInsertDynamicIN2cl4sycl6detail9half_impl4halfELm8ELm16ELN5__spv12MatrixLayoutE0ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEESC_SA_m(%spirv.JointMatrixINTEL._half_8_16_0_3 addrspace(4)*{{.*}}, half %[[#Component]], i64{{.*}})
; CHECK-REGULARIZED: declare dso_local spir_func half @_Z28__spirv_VectorExtractDynamicIN2cl4sycl6detail9half_impl4halfELm8ELm16ELN5__spv12MatrixLayoutE0ELNS5_5Scope4FlagE3EET_PNS5_24__spirv_JointMatrixINTELIS9_XT0_EXT1_EXT2_EXT3_EEEm(%spirv.JointMatrixINTEL._half_8_16_0_3 addrspace(4)* align 2, i64)
; CHECK-REGULARIZED: declare dso_local spir_func %spirv.JointMatrixINTEL._half_8_16_0_3 addrspace(4)* @_Z27__spirv_VectorInsertDynamicIN2cl4sycl6detail9half_impl4halfELm8ELm16ELN5__spv12MatrixLayoutE0ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEESC_SA_m(%spirv.JointMatrixINTEL._half_8_16_0_3 addrspace(4)*, half, i64)

; CHECK-SPIRV: Name [[#VIDValueId:]] "agg.tmp.ascast.ascast"
; CHECK-SPIRV: TypeFloat [[#Float16Id:]] 16
; CHECK-SPIRV: TypeJointMatrixINTEL [[#JointMatrixTyId:]] [[#]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV: VectorExtractDynamic [[#Float16Id]] [[#VEDId:]] [[#]] [[#]]
; CHECK-SPIRV: Store [[#]] [[#VEDId]]
; CHECK-SPIRV: PtrAccessChain [[#]] [[#GEPId:]] [[#VIDValueId]] [[#]] [[#]]
; CHECK-SPIRV: Load [[#Float16Id]] [[#ComponentId:]] [[#GEPId]]
; CHECK-SPIRV: VectorInsertDynamic [[#JointMatrixTyId]] [[#]] [[#]] [[#ComponentId]] [[#]]

; CHECK-LLVM: %[[#ExtractElementCall:]] = call spir_func half @_Z28__spirv_VectorExtractDynamicPU3AS139__spirv_JointMatrixINTEL__half_8_16_0_3l(%spirv.JointMatrixINTEL._half_8_16_0_3 addrspace(1)*{{.*}}, i64{{.*}})
; CHECK-LLVM: %[[#GEP:]] = getelementptr inbounds %"class.cl::sycl::detail::half_impl::half", %"class.cl::sycl::detail::half_impl::half" addrspace(4)*{{.*}}, i32 0, i32 0
; CHECK-LLVM: store half %[[#ExtractElementCall]], half addrspace(4)* %[[#GEP]]

; CHECK-LLVM: %[[#GEP:]] = getelementptr inbounds %"class.cl::sycl::detail::half_impl::half", %"class.cl::sycl::detail::half_impl::half"*{{.*}}, i32 0, i32 0
; CHECK-LLVM: %[[#Component:]] = load half, half* %[[#GEP]]
; CHECK-LLVM: spir_func %spirv.JointMatrixINTEL._half_8_16_0_3 addrspace(1)* @_Z27__spirv_VectorInsertDynamicPU3AS139__spirv_JointMatrixINTEL__half_8_16_0_3Dhl(%spirv.JointMatrixINTEL._half_8_16_0_3 addrspace(1)*{{.*}}, half %[[#Component]], i64{{.*}})

; ModuleID = 'element_wise_all_ops_half.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::detail::half_impl::half" = type { half }
%"struct.cl::sycl::ext::oneapi::experimental::matrix::joint_matrix" = type { %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* }
%"struct.__spv::__spirv_JointMatrixINTEL" = type { [8 x [16 x [1 x [4 x %"class.cl::sycl::detail::half_impl::half"]]]] addrspace(4)* }
%"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" = type { %"struct.cl::sycl::ext::oneapi::experimental::matrix::joint_matrix" addrspace(4)*, i64 }

$_ZN2cl4sycl3ext6oneapi12experimental6matrixplERKNS4_10wi_elementINS0_6detail9half_impl4halfELm8ELm16ELNS4_13matrix_layoutE0ENS2_9sub_groupEEERKS8_ = comdat any

$_ZN2cl4sycl3ext6oneapi12experimental6matrix10wi_elementINS0_6detail9half_impl4halfELm8ELm16ELNS4_13matrix_layoutE0ENS2_9sub_groupEEaSERKS8_ = comdat any

; Function Attrs: convergent mustprogress norecurse
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl3ext6oneapi12experimental6matrixplERKNS4_10wi_elementINS0_6detail9half_impl4halfELm8ELm16ELNS4_13matrix_layoutE0ENS2_9sub_groupEEERKS8_(%"class.cl::sycl::detail::half_impl::half" addrspace(4)* noalias sret(%"class.cl::sycl::detail::half_impl::half") align 2 %agg.result, %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* align 8 dereferenceable(16) %lhs, %"class.cl::sycl::detail::half_impl::half" addrspace(4)* align 2 dereferenceable(2) %rhs) #0 comdat {
entry:
  %lhs.addr = alloca %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)*, align 8
  %ref.tmp1 = alloca %"class.cl::sycl::detail::half_impl::half", align 2
  %lhs.addr.ascast = addrspacecast %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)** %lhs.addr to %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* addrspace(4)*
  %ref.tmp1.ascast = addrspacecast %"class.cl::sycl::detail::half_impl::half"* %ref.tmp1 to %"class.cl::sycl::detail::half_impl::half" addrspace(4)*
  %0 = load %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)*, %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* addrspace(4)* %lhs.addr.ascast, align 8, !tbaa !8
  %M = getelementptr inbounds %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element", %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* %0, i32 0, i32 0
  %1 = load %"struct.cl::sycl::ext::oneapi::experimental::matrix::joint_matrix" addrspace(4)*, %"struct.cl::sycl::ext::oneapi::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %M, align 8, !tbaa !15
  %spvm = getelementptr inbounds %"struct.cl::sycl::ext::oneapi::experimental::matrix::joint_matrix", %"struct.cl::sycl::ext::oneapi::experimental::matrix::joint_matrix" addrspace(4)* %1, i32 0, i32 0
  %2 = load %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)*, %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* addrspace(4)* %spvm, align 8, !tbaa !13
  %3 = load %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)*, %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* addrspace(4)* %lhs.addr.ascast, align 8, !tbaa !8
  %idx = getelementptr inbounds %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element", %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* %3, i32 0, i32 1
  %4 = load i64, i64 addrspace(4)* %idx, align 8, !tbaa !17
  call spir_func void @_Z28__spirv_VectorExtractDynamicIN2cl4sycl6detail9half_impl4halfELm8ELm16ELN5__spv12MatrixLayoutE0ELNS5_5Scope4FlagE3EET_PNS5_24__spirv_JointMatrixINTELIS9_XT0_EXT1_EXT2_EXT3_EEEm(%"class.cl::sycl::detail::half_impl::half" addrspace(4)* sret(%"class.cl::sycl::detail::half_impl::half") align 2 %ref.tmp1.ascast, %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* %2, i64 %4) #2
  ret void
}

; Function Attrs: convergent mustprogress norecurse
define linkonce_odr dso_local spir_func align 8 dereferenceable(16) %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* @_ZN2cl4sycl3ext6oneapi12experimental6matrix10wi_elementINS0_6detail9half_impl4halfELm8ELm16ELNS4_13matrix_layoutE0ENS2_9sub_groupEEaSERKS8_(%"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* align 8 dereferenceable_or_null(16) %this, %"class.cl::sycl::detail::half_impl::half" addrspace(4)* align 2 dereferenceable(2) %rhs) #0 comdat align 2 {
entry:
  %this.addr = alloca %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)*, align 8
  %agg.tmp = alloca %"class.cl::sycl::detail::half_impl::half", align 2
  %this.addr.ascast = addrspacecast %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)** %this.addr to %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* addrspace(4)*
  %agg.tmp.ascast = addrspacecast %"class.cl::sycl::detail::half_impl::half"* %agg.tmp to %"class.cl::sycl::detail::half_impl::half" addrspace(4)*
  %this1 = load %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)*, %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %M = getelementptr inbounds %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element", %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* %this1, i32 0, i32 0
  %0 = load %"struct.cl::sycl::ext::oneapi::experimental::matrix::joint_matrix" addrspace(4)*, %"struct.cl::sycl::ext::oneapi::experimental::matrix::joint_matrix" addrspace(4)* addrspace(4)* %M, align 8, !tbaa !15
  %spvm = getelementptr inbounds %"struct.cl::sycl::ext::oneapi::experimental::matrix::joint_matrix", %"struct.cl::sycl::ext::oneapi::experimental::matrix::joint_matrix" addrspace(4)* %0, i32 0, i32 0
  %1 = load %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)*, %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* addrspace(4)* %spvm, align 8, !tbaa !13
  %idx = getelementptr inbounds %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element", %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* %this1, i32 0, i32 1
  %2 = load i64, i64 addrspace(4)* %idx, align 8, !tbaa !17
  %agg.tmp.ascast.ascast = addrspacecast %"class.cl::sycl::detail::half_impl::half" addrspace(4)* %agg.tmp.ascast to %"class.cl::sycl::detail::half_impl::half"*
  %call = call spir_func %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* @_Z27__spirv_VectorInsertDynamicIN2cl4sycl6detail9half_impl4halfELm8ELm16ELN5__spv12MatrixLayoutE0ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEESC_SA_m(%"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* %1, %"class.cl::sycl::detail::half_impl::half"* byval(%"class.cl::sycl::detail::half_impl::half") align 2 %agg.tmp.ascast.ascast, i64 %2) #2
  ret %"class.cl::sycl::ext::oneapi::experimental::matrix::wi_element" addrspace(4)* %this1
}

; Function Attrs: convergent
declare dso_local spir_func %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* @_Z27__spirv_VectorInsertDynamicIN2cl4sycl6detail9half_impl4halfELm8ELm16ELN5__spv12MatrixLayoutE0ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEESC_SA_m(%"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)*, %"class.cl::sycl::detail::half_impl::half"* byval(%"class.cl::sycl::detail::half_impl::half") align 2, i64) #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z28__spirv_VectorExtractDynamicIN2cl4sycl6detail9half_impl4halfELm8ELm16ELN5__spv12MatrixLayoutE0ELNS5_5Scope4FlagE3EET_PNS5_24__spirv_JointMatrixINTELIS9_XT0_EXT1_EXT2_EXT3_EEEm(%"class.cl::sycl::detail::half_impl::half" addrspace(4)* sret(%"class.cl::sycl::detail::half_impl::half") align 2, %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)*, i64) #1

attributes #0 = { convergent mustprogress norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!opencl.used.extensions = !{!2, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!opencl.used.optional.core.features = !{!4, !3, !3, !4, !3, !4, !3, !3, !3, !4, !3, !4}
!opencl.compiler.options = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!llvm.ident = !{!5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5}
!llvm.module.flags = !{!6, !7}
!sycl.specialization-constants = !{}
!sycl.specialization-constants-default-values = !{}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"cl_khr_fp16"}
!3 = !{}
!4 = !{!"cl_doubles"}
!5 = !{!"Compiler"}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{!"long", !10, i64 0}
!13 = !{!14, !9, i64 0}
!14 = !{!"_ZTSN2cl4sycl3ext6oneapi12experimental6matrix12joint_matrixINS0_6detail9half_impl4halfELm8ELm16ELNS4_13matrix_layoutE0ENS2_9sub_groupEEE", !9, i64 0}
!15 = !{!16, !9, i64 0}
!16 = !{!"_ZTSN2cl4sycl3ext6oneapi12experimental6matrix10wi_elementINS0_6detail9half_impl4halfELm8ELm16ELNS4_13matrix_layoutE0ENS2_9sub_groupEEE", !9, i64 0, !12, i64 8}
!17 = !{!16, !12, i64 8}
