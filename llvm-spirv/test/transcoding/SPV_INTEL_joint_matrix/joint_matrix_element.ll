; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-ext=+all -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability JointMatrixINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_joint_matrix"
; CHECK-SPIRV: TypeInt [[#TypeInt:]] 64
; CHECK-SPIRV: TypeFloat [[#TypeFloat:]] 32
; CHECK-SPIRV: TypeJointMatrixINTEL [[#TypeMatrix:]] [[#TypeFloat]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV: Phi [[#TypeMatrix]] [[#Matrix:]]
; CHECK-SPIRV: JointMatrixWorkItemLengthINTEL [[#TypeInt]] [[#]] [[#Matrix]]
; CHECK-SPIRV: VectorExtractDynamic [[#TypeFloat]] [[#]] [[#Matrix]] [[#Index:]]
; CHECK-SPIRV: FMul [[#TypeFloat]] [[#NewVal:]] [[#]] [[#]]
; CHECK-SPIRV: VectorInsertDynamic [[#TypeMatrix]] [[#]] [[#Matrix]] [[#NewVal]] [[#Index]]

; CHECK-LLVM: [[Length:%.*]] = call spir_func i64 @_Z38__spirv_JointMatrixWorkItemLengthINTELPU3AS141__spirv_JointMatrixINTEL__float_16_16_0_3(%spirv.JointMatrixINTEL._float_16_16_0_3 addrspace(1)* [[Matrix:%.*]])
; CHECK-LLVM: [[Elem:%.*]] = call spir_func float @_Z28__spirv_VectorExtractDynamicPU3AS141__spirv_JointMatrixINTEL__float_16_16_0_3l(%spirv.JointMatrixINTEL._float_16_16_0_3 addrspace(1)* [[Matrix]], i64 [[Index:%.*]])
; CHECK-LLVM: [[NewVal:%.*]] = fmul float [[Elem]], 5.000000e+00
; CHECK-LLVM: {{%.*}} = call spir_func %spirv.JointMatrixINTEL._float_16_16_0_3 addrspace(1)* @_Z27__spirv_VectorInsertDynamicPU3AS141__spirv_JointMatrixINTEL__float_16_16_0_3fl(%spirv.JointMatrixINTEL._float_16_16_0_3 addrspace(1)* [[Matrix]], float [[NewVal]], i64 [[Index]])

source_filename = "/work/tmp/matrix-slice.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"struct.cl::sycl::detail::AssertHappened" = type { i32, [257 x i8], [257 x i8], [129 x i8], i32, i64, i64, i64, i64, i64, i64 }
%"class.cl::sycl::range" = type { %"class.cl::sycl::detail::array" }
%"class.cl::sycl::detail::array" = type { [1 x i64] }
%"class.cl::sycl::id" = type { %"class.cl::sycl::detail::array" }
%"struct.__spv::__spirv_JointMatrixINTEL" = type { [16 x [16 x [1 x [4 x float]]]] addrspace(4)* }

$_ZTSN2cl4sycl6detail23__sycl_service_kernel__16AssertInfoCopierE = comdat any

$_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E6matrix = comdat any

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSN2cl4sycl6detail23__sycl_service_kernel__16AssertInfoCopierE(%"struct.cl::sycl::detail::AssertHappened" addrspace(1)* %_arg_, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %_arg_1, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %_arg_2, %"class.cl::sycl::id"* byval(%"class.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_buffer_location !5 {
entry:
  %0 = getelementptr inbounds %"class.cl::sycl::id", %"class.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds %"struct.cl::sycl::detail::AssertHappened", %"struct.cl::sycl::detail::AssertHappened" addrspace(1)* %_arg_, i64 %2
  %3 = bitcast %"struct.cl::sycl::detail::AssertHappened" addrspace(1)* %add.ptr.i to i8 addrspace(1)*
  %4 = addrspacecast i8 addrspace(1)* %3 to i8 addrspace(4)*
  tail call spir_func void @__devicelib_assert_read(i8 addrspace(4)* %4) #2
  ret void
}

; Function Attrs: convergent
declare extern_weak dso_local spir_func void @__devicelib_assert_read(i8 addrspace(4)*) local_unnamed_addr #1

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_E6matrix() local_unnamed_addr #0 comdat !kernel_arg_buffer_location !6 {
entry:
  %call9.i.i = tail call spir_func %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIfLm16ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(float addrspace(4)* addrspacecast (float addrspace(1)* null to float addrspace(4)*), i64 1, i32 0, i32 3, i32 0) #2
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.body.i, %entry
  %A.sroa.0.0.i = phi %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* [ %call9.i.i, %entry ], [ %call5.i.i, %for.body.i ]
  %i.0.i = phi i32 [ 0, %entry ], [ %inc.i, %for.body.i ]
  %conv.i = zext i32 %i.0.i to i64
  %call.i12.i = tail call spir_func i64 @_Z38__spirv_JointMatrixWorkItemLengthINTELIfLm16ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEmPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEE(%"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* %A.sroa.0.0.i) #2
  %cmp.i = icmp ugt i64 %call.i12.i, %conv.i
  br i1 %cmp.i, label %for.body.i, label %_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlNS0_7nd_itemILi2EEEE_clES5_.exit

for.body.i:                                       ; preds = %for.cond.i
  %call.i.i = tail call spir_func float @_Z28__spirv_VectorExtractDynamicIfLm16ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EmET_PNS0_24__spirv_JointMatrixINTELIS4_XT0_EXT1_EXT2_EXT3_EEET4_(%"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* %A.sroa.0.0.i, i64 %conv.i) #2
  %mul.i.i = fmul float %call.i.i, 5.000000e+00
  %call5.i.i = tail call spir_func %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* @_Z27__spirv_VectorInsertDynamicIfLm16ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EmEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEES7_T4_S5_(%"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* %A.sroa.0.0.i, float %mul.i.i, i64 %conv.i) #2
  %inc.i = add nuw nsw i32 %i.0.i, 1
  br label %for.cond.i, !llvm.loop !7

_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlNS0_7nd_itemILi2EEEE_clES5_.exit: ; preds = %for.cond.i
  tail call spir_func void @_Z29__spirv_JointMatrixStoreINTELIfLm16ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEvPT_PNS0_24__spirv_JointMatrixINTELIS4_XT0_EXT1_EXT2_EXT3_EEEmS1_S3_i(float addrspace(4)* addrspacecast (float addrspace(1)* null to float addrspace(4)*), %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* %A.sroa.0.0.i, i64 1, i32 0, i32 3, i32 0) #2
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIfLm16ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEEPS5_mS1_S3_i(float addrspace(4)*, i64, i32, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func i64 @_Z38__spirv_JointMatrixWorkItemLengthINTELIfLm16ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEmPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEE(%"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)*) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func float @_Z28__spirv_VectorExtractDynamicIfLm16ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EmET_PNS0_24__spirv_JointMatrixINTELIS4_XT0_EXT1_EXT2_EXT3_EEET4_(%"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)*, i64) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)* @_Z27__spirv_VectorInsertDynamicIfLm16ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EmEPNS0_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEES7_T4_S5_(%"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)*, float, i64) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z29__spirv_JointMatrixStoreINTELIfLm16ELm16ELN5__spv12MatrixLayoutE0ELNS0_5Scope4FlagE3EEvPT_PNS0_24__spirv_JointMatrixINTELIS4_XT0_EXT1_EXT2_EXT3_EEEmS1_S3_i(float addrspace(4)*, %"struct.__spv::__spirv_JointMatrixINTEL" addrspace(4)*, i64, i32, i32, i32) local_unnamed_addr #1

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/work/tmp/matrix-slice.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 14.0.0 (https://github.com/intel/llvm.git 3648adf79e4fdb619fdbe41d63bc39f456b5be8c)"}
!5 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!6 = !{}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.mustprogress"}
