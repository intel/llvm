; RUN: llvm-as < %s -o %t.bc

; RUN: llvm-spirv -s %t.bc -o %t.regularized.bc
; RUN: llvm-dis %t.regularized.bc -o %t.regularized.ll
; RUN: FileCheck < %t.regularized.ll %s --check-prefix=CHECK-REGULARIZED

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_bfloat16_conversion,+SPV_INTEL_joint_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-REGULARIZED: %[[Alloca:.*]] = alloca %"class.cl::sycl::ext::intel::experimental::bfloat16", align 2
; CHECK-REGULARIZED: %[[ASCast:.*]] = addrspacecast %"class.cl::sycl::ext::intel::experimental::bfloat16"* %[[Alloca]] to %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)*
; CHECK-REGULARIZED: %[[GEP1:.*]] = getelementptr inbounds %"class.cl::sycl::ext::intel::experimental::bfloat16", %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)* %[[ASCast]], i64 0, i32 0
; CHECK-REGULARIZED: %[[#Extract:]] = call spir_func i16 @_Z28__spirv_VectorExtractDynamicIN2cl4sycl3ext5intel12experimental8bfloat16ELm8ELm16ELN5__spv12MatrixLayoutE0ELNS6_5Scope4FlagE3EET_PNS6_24__spirv_JointMatrixINTELISA_XT0_EXT1_EXT2_EXT3_EEEm(%spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* align 2 %{{.*}}, i64 noundef %{{.*}})
; CHECK-REGULARIZED: %[[#GEP2:]] = getelementptr inbounds %"class.cl::sycl::ext::intel::experimental::bfloat16", %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)* %[[ASCast]], i32 0, i32 0
; CHECK-REGULARIZED: store i16 %[[#Extract]], i16 addrspace(4)* %[[#GEP2]], align 2
; CHECK-REGULARIZED: %[[#Load:]] = load i16, i16 addrspace(4)* %[[GEP1]], align 2
; CHECK-REGULARIZED: %[[ConvertVal:.*]] = call spir_func noundef float @_Z27__spirv_ConvertBF16ToFINTELt(i16 noundef zeroext %[[#Load]])
; CHECK-REGULARIZED: %{{.*}} = fadd float %[[ConvertVal]], %{{.*}}

; CHECK-SPIRV: TypeInt [[#TypeI16ID:]] 16 0
; CHECK-SPIRV: TypeFloat [[#TypeFID:]] 32
; CHECK-SPIRV: TypeJointMatrixINTEL [[#TypeJointMID:]] [[#TypeI16ID]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV: Phi [[#TypeJointMID]] [[#PhiID:]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV: VectorExtractDynamic [[#TypeI16ID]] [[#ExtractID:]] [[#PhiID]] [[#]]
; CHECK-SPIRV: Store [[#PtrID:]] [[#ExtractID]] [[#]] [[#]]
; CHECK-SPIRV: ConvertBF16ToFINTEL [[#TypeFID]] [[#Conv1ID:]] [[#]]
; CHECK-SPIRV: ConvertBF16ToFINTEL [[#TypeFID]] [[#Conv2ID:]] [[#]]
; CHECK-SPIRV: FAdd [[#TypeFID]] [[#ResId:]] [[#Conv1ID]] [[#Conv2ID]]
; CHECK-SPIRV: ConvertFToBF16INTEL [[#TypeI16ID]] [[#]] [[#ResId]]
; CHECK-SPIRV: Load [[#TypeI16ID]] [[#LoadID:]] [[#]] [[#]] [[#]]
; CHECK-SPIRV: VectorInsertDynamic [[#TypeJointMID]] [[#]] [[#PhiID]] [[#LoadID]] [[#]]

; CHECK-LLVM: %spirv.JointMatrixINTEL._short_8_16_0_3
; CHECK-LLVM: %[[GEP1:.*]] = getelementptr inbounds %"class.cl::sycl::ext::intel::experimental::bfloat16", %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)* %{{.*}}, i64 0, i32 0
; CHECK-LLVM: %[[GEP2:.*]] = getelementptr inbounds %"class.cl::sycl::ext::intel::experimental::bfloat16", %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)* %{{.*}}, i64 0, i32 0
; CHECK-LLVM: %[[ConvertConst:.*]] = call spir_func i16 @_Z32intel_convert_bfloat16_as_ushortf(float 2.000000e+00)
; CHECK-LLVM: %[[#LoadGEP:]] = load i16, i16 addrspace(4)* %[[GEP2]], align 2
; CHECK-LLVM: %[[ConvertVal:.*]] = call spir_func float @_Z31intel_convert_as_bfloat16_floats(i16 %[[#LoadGEP]])
; CHECK-LLVM: %[[ConvertConstToF:.*]] = call spir_func float @_Z31intel_convert_as_bfloat16_floats(i16 %[[ConvertConst]])
; CHECK-LLVM: %[[FAddRes:.*]] = fadd float %[[ConvertVal]], %[[ConvertConstToF]]
; CHECK-LLVM: %[[ConvertResToBF:.*]] = call spir_func i16 @_Z32intel_convert_bfloat16_as_ushortf(float %[[FAddRes]])
; CHECK-LLVM: store i16 %[[ConvertResToBF]], i16 addrspace(4)* %[[#]], align 2

; ModuleID = 'joint_matrix_bfloat16_test.bc'
source_filename = "joint_matrix_bfloat16_test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class.anon = type { %"class.cl::sycl::accessor" }
%"class.cl::sycl::accessor" = type { %"class.cl::sycl::detail::AccessorImplDevice", %union.anon }
%"class.cl::sycl::detail::AccessorImplDevice" = type { %"class.cl::sycl::id", %"class.cl::sycl::range", %"class.cl::sycl::range" }
%"class.cl::sycl::id" = type { %"class.cl::sycl::detail::array" }
%"class.cl::sycl::detail::array" = type { [2 x i64] }
%"class.cl::sycl::range" = type { %"class.cl::sycl::detail::array" }
%union.anon = type { %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(1)* }
%"class.cl::sycl::ext::intel::experimental::bfloat16" = type { i16 }
%"class.cl::sycl::nd_item" = type { %"class.cl::sycl::item", %"class.cl::sycl::item.0", %"class.cl::sycl::group" }
%"class.cl::sycl::item" = type { %"struct.cl::sycl::detail::ItemBase" }
%"struct.cl::sycl::detail::ItemBase" = type { %"class.cl::sycl::range", %"class.cl::sycl::id", %"class.cl::sycl::id" }
%"class.cl::sycl::item.0" = type { %"struct.cl::sycl::detail::ItemBase.1" }
%"struct.cl::sycl::detail::ItemBase.1" = type { %"class.cl::sycl::range", %"class.cl::sycl::id" }
%"class.cl::sycl::group" = type { %"class.cl::sycl::range", %"class.cl::sycl::range", %"class.cl::sycl::range", %"class.cl::sycl::id" }
%spirv.JointMatrixINTEL._bfloat16_8_16_0_3 = type opaque

$_ZZZ17matrix_verify_addIN2cl4sycl3ext5intel12experimental8bfloat16ELm16ELm16EEvNS1_5queueER10big_matrixIT_XT0_EXT1_EERNS1_8nd_rangeILi2EEEfENKUlRNS1_7handlerEE_clESF_ENKUlNS1_7nd_itemILi2EEEE_clESI_ = comdat any

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: convergent inlinehint norecurse
define linkonce_odr dso_local spir_func void @_ZZZ17matrix_verify_addIN2cl4sycl3ext5intel12experimental8bfloat16ELm16ELm16EEvNS1_5queueER10big_matrixIT_XT0_EXT1_EERNS1_8nd_rangeILi2EEEfENKUlRNS1_7handlerEE_clESF_ENKUlNS1_7nd_itemILi2EEEE_clESI_(%class.anon addrspace(4)* noundef align 8 dereferenceable_or_null(56) %this, %"class.cl::sycl::nd_item"* noundef byval(%"class.cl::sycl::nd_item") align 8 %spmd_item) local_unnamed_addr #1 comdat align 2 {
entry:
  %ref.tmp.i = alloca %"class.cl::sycl::ext::intel::experimental::bfloat16", align 2
  %agg.tmp.i54 = alloca %"class.cl::sycl::ext::intel::experimental::bfloat16", align 2
  %agg.tmp.i = alloca %"class.cl::sycl::ext::intel::experimental::bfloat16", align 2
  %spmd_item.ascast = addrspacecast %"class.cl::sycl::nd_item"* %spmd_item to %"class.cl::sycl::nd_item" addrspace(4)*
  %arrayidx.i.i.i = getelementptr inbounds %"class.cl::sycl::nd_item", %"class.cl::sycl::nd_item" addrspace(4)* %spmd_item.ascast, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i64 0
  %0 = load i64, i64 addrspace(4)* %arrayidx.i.i.i, align 8, !tbaa !5
  %cmp.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i)
  %arrayidx.i.i.i29 = getelementptr inbounds %"class.cl::sycl::nd_item", %"class.cl::sycl::nd_item" addrspace(4)* %spmd_item.ascast, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i64 1
  %1 = load i64, i64 addrspace(4)* %arrayidx.i.i.i29, align 8, !tbaa !5
  %cmp.i30 = icmp ult i64 %1, 2147483648
  tail call void @llvm.assume(i1 %cmp.i30)
  %arrayidx.i.i.i31 = getelementptr inbounds %"class.cl::sycl::nd_item", %"class.cl::sycl::nd_item" addrspace(4)* %spmd_item.ascast, i64 0, i32 1, i32 0, i32 1, i32 0, i32 0, i64 0
  %2 = load i64, i64 addrspace(4)* %arrayidx.i.i.i31, align 8, !tbaa !5
  %cmp.i32 = icmp ult i64 %2, 2147483648
  tail call void @llvm.assume(i1 %cmp.i32)
  %arrayidx.i.i.i33 = getelementptr inbounds %"class.cl::sycl::nd_item", %"class.cl::sycl::nd_item" addrspace(4)* %spmd_item.ascast, i64 0, i32 1, i32 0, i32 1, i32 0, i32 0, i64 1
  %3 = load i64, i64 addrspace(4)* %arrayidx.i.i.i33, align 8, !tbaa !5
  %cmp.i34 = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i34)
  %4 = bitcast %"class.cl::sycl::ext::intel::experimental::bfloat16"* %agg.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %4)
  %agg.tmp.ascast.i = addrspacecast %"class.cl::sycl::ext::intel::experimental::bfloat16"* %agg.tmp.i to %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)*
  %call.i.i.i = tail call spir_func noundef zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float noundef 5.000000e+00) #6
  %value.i.i = getelementptr inbounds %"class.cl::sycl::ext::intel::experimental::bfloat16", %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)* %agg.tmp.ascast.i, i64 0, i32 0
  store i16 %call.i.i.i, i16 addrspace(4)* %value.i.i, align 2, !tbaa !9
  %call.i = tail call spir_func noundef %spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* @_Z26__spirv_CompositeConstructIN2cl4sycl3ext5intel12experimental8bfloat16ELm8ELm16ELN5__spv12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEESB_(%"class.cl::sycl::ext::intel::experimental::bfloat16"* noundef nonnull byval(%"class.cl::sycl::ext::intel::experimental::bfloat16") align 2 %agg.tmp.i) #7
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %4)
  %ref.tmp.ascast.i = addrspacecast %"class.cl::sycl::ext::intel::experimental::bfloat16"* %ref.tmp.i to %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)*
  %5 = bitcast %"class.cl::sycl::ext::intel::experimental::bfloat16"* %ref.tmp.i to i8*
  %value.i.i.i = getelementptr inbounds %"class.cl::sycl::ext::intel::experimental::bfloat16", %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)* %ref.tmp.ascast.i, i64 0, i32 0
  %6 = bitcast %"class.cl::sycl::ext::intel::experimental::bfloat16"* %agg.tmp.i54 to i8*
  %7 = getelementptr inbounds %"class.cl::sycl::ext::intel::experimental::bfloat16", %"class.cl::sycl::ext::intel::experimental::bfloat16"* %agg.tmp.i54, i64 0, i32 0
  %8 = addrspacecast i16* %7 to i16 addrspace(4)*
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %sub_a.sroa.0.0 = phi %spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* [ %call.i, %entry ], [ %call.i58, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %conv = zext i32 %i.0 to i64
  %call.i41 = call spir_func noundef i64 @_Z38__spirv_JointMatrixWorkItemLengthINTELIN2cl4sycl3ext5intel12experimental8bfloat16ELm8ELm16ELN5__spv12MatrixLayoutE0ELNS6_5Scope4FlagE3EEmPNS6_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEE(%spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* noundef %sub_a.sroa.0.0) #7
  %cmp = icmp ugt i64 %call.i41, %conv
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %sub5 = sub nsw i64 %1, %3
  %sub = sub nsw i64 %0, %2
  %MData.i.i.i = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this, i64 0, i32 0, i32 1, i32 0
  %9 = load %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(1)*, %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(1)* addrspace(4)* %MData.i.i.i, align 8, !tbaa !12, !noalias !13
  %mul19 = shl nsw i64 %sub, 7
  %add.ptr.i = getelementptr inbounds %"class.cl::sycl::ext::intel::experimental::bfloat16", %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(1)* %9, i64 %mul19
  %div = and i64 %sub5, -8
  %add.ptr.i45 = getelementptr inbounds %"class.cl::sycl::ext::intel::experimental::bfloat16", %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(1)* %add.ptr.i, i64 %div
  %call.ascast.i = addrspacecast %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(1)* %add.ptr.i45 to %"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)*
  call spir_func void @_Z29__spirv_JointMatrixStoreINTELIN2cl4sycl3ext5intel12experimental8bfloat16ELm8ELm16ELN5__spv12MatrixLayoutE0ELNS6_5Scope4FlagE3EEvPT_PNS6_24__spirv_JointMatrixINTELISA_XT0_EXT1_EXT2_EXT3_EEEmS7_S9_i(%"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)* noundef %call.ascast.i, %spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* noundef %sub_a.sroa.0.0, i64 noundef 16, i32 noundef 0, i32 noundef 3, i32 noundef 0) #7
  ret void

for.body:                                         ; preds = %for.cond
  %call.i.i = call spir_func noundef zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float noundef 2.000000e+00) #6
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %5) #8, !noalias !16
  call spir_func void @_Z28__spirv_VectorExtractDynamicIN2cl4sycl3ext5intel12experimental8bfloat16ELm8ELm16ELN5__spv12MatrixLayoutE0ELNS6_5Scope4FlagE3EET_PNS6_24__spirv_JointMatrixINTELISA_XT0_EXT1_EXT2_EXT3_EEEm(%"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)* sret(%"class.cl::sycl::ext::intel::experimental::bfloat16") align 2 %ref.tmp.ascast.i, %spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* noundef %sub_a.sroa.0.0, i64 noundef %conv) #7, !noalias !16
  %10 = load i16, i16 addrspace(4)* %value.i.i.i, align 2, !tbaa !19, !noalias !20
  %call.i.i.i.i = call spir_func noundef float @_Z27__spirv_ConvertBF16ToFINTELt(i16 noundef zeroext %10) #6, !noalias !20
  %call.i.i3.i.i = call spir_func noundef float @_Z27__spirv_ConvertBF16ToFINTELt(i16 noundef zeroext %call.i.i) #6, !noalias !20
  %add.i.i = fadd float %call.i.i.i.i, %call.i.i3.i.i
  %call.i.i4.i.i = call spir_func noundef zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float noundef %add.i.i) #6, !noalias !20
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %5) #8, !noalias !16
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %6)
  store i16 %call.i.i4.i.i, i16 addrspace(4)* %8, align 2, !tbaa !19
  %call.i58 = call spir_func noundef %spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* @_Z27__spirv_VectorInsertDynamicIN2cl4sycl3ext5intel12experimental8bfloat16ELm8ELm16ELN5__spv12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEESD_SB_m(%spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* noundef %sub_a.sroa.0.0, %"class.cl::sycl::ext::intel::experimental::bfloat16"* noundef nonnull byval(%"class.cl::sycl::ext::intel::experimental::bfloat16") align 2 %agg.tmp.i54, i64 noundef %conv) #7
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %6)
  %inc = add nuw nsw i32 %i.0, 1
  br label %for.cond, !llvm.loop !23
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: inaccessiblememonly nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #3

; Function Attrs: convergent
declare dso_local spir_func noundef %spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* @_Z26__spirv_CompositeConstructIN2cl4sycl3ext5intel12experimental8bfloat16ELm8ELm16ELN5__spv12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEESB_(%"class.cl::sycl::ext::intel::experimental::bfloat16"* noundef byval(%"class.cl::sycl::ext::intel::experimental::bfloat16") align 2) local_unnamed_addr #4

; Function Attrs: convergent
declare dso_local spir_func noundef i64 @_Z38__spirv_JointMatrixWorkItemLengthINTELIN2cl4sycl3ext5intel12experimental8bfloat16ELm8ELm16ELN5__spv12MatrixLayoutE0ELNS6_5Scope4FlagE3EEmPNS6_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEE(%spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* noundef) local_unnamed_addr #4

; Function Attrs: convergent
declare dso_local spir_func void @_Z28__spirv_VectorExtractDynamicIN2cl4sycl3ext5intel12experimental8bfloat16ELm8ELm16ELN5__spv12MatrixLayoutE0ELNS6_5Scope4FlagE3EET_PNS6_24__spirv_JointMatrixINTELISA_XT0_EXT1_EXT2_EXT3_EEEm(%"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)* sret(%"class.cl::sycl::ext::intel::experimental::bfloat16") align 2, %spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* noundef, i64 noundef) local_unnamed_addr #4

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef float @_Z27__spirv_ConvertBF16ToFINTELt(i16 noundef zeroext) local_unnamed_addr #5

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float noundef) local_unnamed_addr #5

; Function Attrs: convergent
declare dso_local spir_func noundef %spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* @_Z27__spirv_VectorInsertDynamicIN2cl4sycl3ext5intel12experimental8bfloat16ELm8ELm16ELN5__spv12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT2_EXT3_EEESD_SB_m(%spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* noundef, %"class.cl::sycl::ext::intel::experimental::bfloat16"* noundef byval(%"class.cl::sycl::ext::intel::experimental::bfloat16") align 2, i64 noundef) local_unnamed_addr #4

; Function Attrs: convergent
declare dso_local spir_func void @_Z29__spirv_JointMatrixStoreINTELIN2cl4sycl3ext5intel12experimental8bfloat16ELm8ELm16ELN5__spv12MatrixLayoutE0ELNS6_5Scope4FlagE3EEvPT_PNS6_24__spirv_JointMatrixINTELISA_XT0_EXT1_EXT2_EXT3_EEEmS7_S9_i(%"class.cl::sycl::ext::intel::experimental::bfloat16" addrspace(4)* noundef, %spirv.JointMatrixINTEL._bfloat16_8_16_0_3 addrspace(4)* noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #4

attributes #0 = { argmemonly nofree nosync nounwind willreturn }
attributes #1 = { convergent inlinehint norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { argmemonly nofree nounwind willreturn writeonly }
attributes #3 = { inaccessiblememonly nofree nosync nounwind willreturn }
attributes #4 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #5 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #6 = { convergent nounwind }
attributes #7 = { convergent }
attributes #8 = { nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 15.0.0 (https://github.com/pauzinl/llvm.git fb27c655023f19ff91f09413a0c51f0a37071cff)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"long", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !11, i64 0}
!10 = !{!"_ZTSN2cl4sycl3ext5intel12experimental8bfloat16E", !11, i64 0}
!11 = !{!"short", !7, i64 0}
!12 = !{!7, !7, i64 0}
!13 = !{!14}
!14 = distinct !{!14, !15, !"_ZNK2cl4sycl8accessorINS0_3ext5intel12experimental8bfloat16ELi2ELNS0_6access4modeE1026ELNS6_6targetE2014ELNS6_11placeholderE0ENS2_6oneapi22accessor_property_listIJEEEE11get_pointerILS8_2014EvEENS0_9multi_ptrIS5_LNS6_13address_spaceE1EEEv: %agg.result"}
!15 = distinct !{!15, !"_ZNK2cl4sycl8accessorINS0_3ext5intel12experimental8bfloat16ELi2ELNS0_6access4modeE1026ELNS6_6targetE2014ELNS6_11placeholderE0ENS2_6oneapi22accessor_property_listIJEEEE11get_pointerILS8_2014EvEENS0_9multi_ptrIS5_LNS6_13address_spaceE1EEEv"}
!16 = !{!17}
!17 = distinct !{!17, !18, !"_ZN2cl4sycl3ext6oneapi12experimental6matrixplERKNS4_10wi_elementINS1_5intel12experimental8bfloat16ELm8ELm16ELNS4_13matrix_layoutE0ENS2_9sub_groupEEERKS8_: %agg.result"}
!18 = distinct !{!18, !"_ZN2cl4sycl3ext6oneapi12experimental6matrixplERKNS4_10wi_elementINS1_5intel12experimental8bfloat16ELm8ELm16ELNS4_13matrix_layoutE0ENS2_9sub_groupEEERKS8_"}
!19 = !{!11, !11, i64 0}
!20 = !{!21, !17}
!21 = distinct !{!21, !22, !"_ZN2cl4sycl3ext5intel12experimentalplERKNS3_8bfloat16ES6_: %agg.result"}
!22 = distinct !{!22, !"_ZN2cl4sycl3ext5intel12experimentalplERKNS3_8bfloat16ES6_"}
!23 = distinct !{!23, !24}
!24 = !{!"llvm.loop.mustprogress"}
