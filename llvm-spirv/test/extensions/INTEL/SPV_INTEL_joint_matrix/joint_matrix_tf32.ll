; RUN: llvm-as -opaque-pointers=0 < %s -o %t.bc

; RUN: llvm-spirv %t.bc -opaque-pointers=0 --spirv-ext=+SPV_INTEL_tensor_float32_conversion,+SPV_INTEL_joint_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis -opaque-pointers=0 < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability TensorFloat32ConversionINTEL
; CHECK-SPIRV-DAG: Capability JointMatrixINTEL
; CHECK-SPIRV-DAG: Capability JointMatrixTF32ComponentTypeINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_tensor_float32_conversion"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_joint_matrix"
; CHECK-SPIRV-DAG: TypeInt [[#TypeInt:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#TypeInt]] [[#CTI:]] 1 {{$}}
; CHECK-SPIRV-DAG: TypeFloat [[#FloatTy:]] 32
; CHECK-SPIRV: TypeJointMatrixINTEL [[#]] [[#FloatTy]] [[#]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV: TypeJointMatrixINTEL [[#]] [[#FloatTy]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#CTI]]
; CHECK-SPIRV: TypeJointMatrixINTEL [[#]] [[#FloatTy]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#CTI]]

; CHECK-LLVM: %spirv.JointMatrixINTEL._float_8_16_3_3_2 = type opaque
; CHECK-LLVM: %spirv.JointMatrixINTEL._tf32_8_16_0_3_0 = type opaque
; CHECK-LLVM: %spirv.JointMatrixINTEL._tf32_16_16_0_3_1 = type opaque

; ModuleID = 'matrix-tf32-test-sycl-spir64-unknown-unknown.bc'
source_filename = "matrix-tf32-test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%spirv.JointMatrixINTEL._float_8_16_3_3_2 = type opaque
%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 = type opaque
%spirv.JointMatrixINTEL._tf32_16_16_0_3_1 = type opaque

$_ZTSZZ15matrix_multiplyIffLm16ELm32ELm32ELm32ELm16ELm32EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN4sycl3_V17handlerEE_clESC_E7imatrix = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiplyIffLm16ELm32ELm32ELm32ELm16ELm32EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN4sycl3_V17handlerEE_clESC_E7imatrix(float addrspace(1)* noundef align 4 %_arg_accC, i64 noundef %_arg_N, i64 noundef %_arg_K, float addrspace(1)* noundef align 4 %_arg_accA, float addrspace(1)* noundef align 4 %_arg_accB, %"class.sycl::_V1::range"* noundef byval(%"class.sycl::_V1::range") align 8 %_arg_accB8, %"class.sycl::_V1::id"* noundef byval(%"class.sycl::_V1::id") align 8 %_arg_accB9) local_unnamed_addr #0 {
entry:
  %agg.tmp19.sroa.0.sroa.2.0..sroa_idx = getelementptr inbounds %"class.sycl::_V1::range", %"class.sycl::_V1::range"* %_arg_accB8, i64 0, i32 0, i32 0, i64 1
  %agg.tmp19.sroa.0.sroa.2.0.copyload = load i64, i64* %agg.tmp19.sroa.0.sroa.2.0..sroa_idx, align 8
  %0 = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %_arg_accB9, i64 0, i32 0, i32 0, i64 0
  %agg.tmp20.sroa.0.sroa.0.0.copyload = load i64, i64* %0, align 8
  %agg.tmp20.sroa.0.sroa.2.0..sroa_idx = getelementptr inbounds %"class.sycl::_V1::id", %"class.sycl::_V1::id"* %_arg_accB9, i64 0, i32 0, i32 0, i64 1
  %agg.tmp20.sroa.0.sroa.2.0.copyload = load i64, i64* %agg.tmp20.sroa.0.sroa.2.0..sroa_idx, align 8
  %mul.i4.i.i.i.i67 = mul i64 %agg.tmp20.sroa.0.sroa.0.0.copyload, %agg.tmp19.sroa.0.sroa.2.0.copyload
  %add.i6.i.i.i.i68 = add i64 %mul.i4.i.i.i.i67, %agg.tmp20.sroa.0.sroa.2.0.copyload
  %add.ptr.i69 = getelementptr inbounds float, float addrspace(1)* %_arg_accB, i64 %add.i6.i.i.i.i68
  %1 = load <3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, align 32
  %2 = extractelement <3 x i64> %1, i64 1
  %3 = extractelement <3 x i64> %1, i64 0
  %4 = load <3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId, align 32
  %5 = extractelement <3 x i64> %4, i64 1
  %6 = extractelement <3 x i64> %4, i64 0
  %cmp.i.i = icmp ult i64 %2, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %cmp.i136.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i136.i)
  %cmp.i138.i = icmp ult i64 %5, 2147483648
  tail call void @llvm.assume(i1 %cmp.i138.i)
  %sub.i = sub nsw i64 %2, %5
  %cmp.i140.i = icmp ult i64 %6, 2147483648
  tail call void @llvm.assume(i1 %cmp.i140.i)
  %sub5.i = sub nsw i64 %3, %6
  %mul.i = shl nsw i64 %sub.i, 3
  %mul8.i = mul i64 %mul.i, %_arg_N
  %add.ptr.i.i = getelementptr inbounds float, float addrspace(1)* %_arg_accC, i64 %mul8.i
  %div134.i = and i64 %sub5.i, -16
  %add.ptr.i182.i = getelementptr inbounds float, float addrspace(1)* %add.ptr.i.i, i64 %div134.i
  %call.ascast.i.i = addrspacecast float addrspace(1)* %add.ptr.i182.i to float addrspace(4)*
  %call1.i.i = tail call spir_func noundef %spirv.JointMatrixINTEL._float_8_16_3_3_2 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIffLm8ELm16ELN5__spv9MatrixUseE2ELNS0_12MatrixLayoutE3ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS2_S4_i(float addrspace(4)* noundef %call.ascast.i.i, i64 noundef %_arg_N, i32 noundef 0, i32 noundef 3, i32 noundef 0) #3
  %mul17.i = mul i64 %mul.i, %_arg_K
  %add.ptr.i194.i = getelementptr inbounds float, float addrspace(1)* %_arg_accA, i64 %mul17.i
  %idx.neg.i.i205.i = sub i64 0, %add.i6.i.i.i.i68
  %add.ptr.i.i206334.i = getelementptr float, float addrspace(1)* %add.ptr.i69, i64 %div134.i
  %add.ptr.i209333.i = getelementptr float, float addrspace(1)* %add.ptr.i.i206334.i, i64 %idx.neg.i.i205.i
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.cond.cleanup58.i, %entry
  %sub_a.sroa.0.0.i = phi %spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* [ undef, %entry ], [ %sub_a.sroa.0.1.i, %for.cond.cleanup58.i ]
  %sub_c.sroa.0.0.i = phi %spirv.JointMatrixINTEL._float_8_16_3_3_2 addrspace(4)* [ %call1.i.i, %entry ], [ %call.i168.i, %for.cond.cleanup58.i ]
  %k.0.i = phi i32 [ 0, %entry ], [ %add.i, %for.cond.cleanup58.i ]
  %conv.i = zext i32 %k.0.i to i64
  %cmp.i = icmp ult i64 %conv.i, %_arg_K
  br i1 %cmp.i, label %for.body.i, label %for.cond82.i

for.body.i:                                       ; preds = %for.cond.i
  %add.ptr.i197.i = getelementptr inbounds float, float addrspace(1)* %add.ptr.i194.i, i64 %conv.i
  %call.ascast.i148.i = addrspacecast float addrspace(1)* %add.ptr.i197.i to float addrspace(4)*
  %call1.i149.i = tail call spir_func noundef %spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEPNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mSA_SC_i(float addrspace(4)* noundef %call.ascast.i148.i, i64 noundef %_arg_K, i32 noundef 0, i32 noundef 3, i32 noundef 0) #3
  %mul26.i = mul i64 %conv.i, %_arg_N
  %add.ptr.i212.i = getelementptr float, float addrspace(1)* %add.ptr.i209333.i, i64 %mul26.i
  %call.ascast.i155.i = addrspacecast float addrspace(1)* %add.ptr.i212.i to float addrspace(4)*
  %call1.i156.i = tail call spir_func noundef %spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm16ELm16ELN5__spv9MatrixUseE1ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEPNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mSA_SC_i(float addrspace(4)* noundef %call.ascast.i155.i, i64 noundef %_arg_N, i32 noundef 0, i32 noundef 3, i32 noundef 0) #3
  br label %for.cond30.i

for.cond30.i:                                     ; preds = %for.body37.i, %for.body.i
  %sub_a.sroa.0.1.i = phi %spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* [ %call1.i149.i, %for.body.i ], [ %call.i225.i, %for.body37.i ]
  %i.0.i = phi i32 [ 0, %for.body.i ], [ %inc.i, %for.body37.i ]
  %conv31.i = zext i32 %i.0.i to i64
  %call.i215.i = tail call spir_func noundef i64 @_Z38__spirv_JointMatrixWorkItemLengthINTELIN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEmPNS8_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT3_EXT4_EXT2_EEE(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef %sub_a.sroa.0.1.i) #3
  %cmp35.i = icmp ugt i64 %call.i215.i, %conv31.i
  br i1 %cmp35.i, label %for.body37.i, label %for.cond52.i

for.body37.i:                                     ; preds = %for.cond30.i
  %call.i218.i = tail call spir_func noundef float @_Z28__spirv_VectorExtractDynamicIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EET_PNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEm(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef %sub_a.sroa.0.1.i, i64 noundef %conv31.i) #3
  %call.i.i = tail call spir_func noundef float @_Z27__spirv_ConvertFToTF32INTELf(float noundef %call.i218.i) #3
  %call.i225.i = tail call spir_func noundef %spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* @_Z27__spirv_VectorInsertDynamicIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEPNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEESG_T_m(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef %sub_a.sroa.0.1.i, float noundef %call.i.i, i64 noundef %conv31.i) #3
  %inc.i = add nuw nsw i32 %i.0.i, 1
  br label %for.cond30.i

for.cond52.i:                                     ; preds = %for.cond30.i, %for.body59.i
  %sub_b.sroa.0.0.i = phi %spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* [ %call.i243.i, %for.body59.i ], [ %call1.i156.i, %for.cond30.i ]
  %i51.0.i = phi i32 [ %inc74.i, %for.body59.i ], [ 0, %for.cond30.i ]
  %conv53.i = zext i32 %i51.0.i to i64
  %call.i229.i = tail call spir_func noundef i64 @_Z38__spirv_JointMatrixWorkItemLengthINTELIN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm16ELm16ELN5__spv9MatrixUseE1ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEmPNS8_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT3_EXT4_EXT2_EEE(%spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* noundef %sub_b.sroa.0.0.i) #3
  %cmp57.i = icmp ugt i64 %call.i229.i, %conv53.i
  br i1 %cmp57.i, label %for.body59.i, label %for.cond.cleanup58.i

for.cond.cleanup58.i:                             ; preds = %for.cond52.i
  %call.i168.i = tail call spir_func noundef %spirv.JointMatrixINTEL._float_8_16_3_3_2 addrspace(4)* @_Z27__spirv_JointMatrixMadINTELIN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32EfLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS9_1ELS9_2ELNS8_12MatrixLayoutE0ELSA_0ELSA_3ELNS8_5Scope4FlagE3EEPNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT9_EXT10_EXT6_EEEPNSD_IT_XT1_EXT2_EXT7_EXT10_EXT4_EEEPNSD_ISH_XT2_EXT3_EXT8_EXT10_EXT5_EEESG_SC_(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef %sub_a.sroa.0.1.i, %spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* noundef %sub_b.sroa.0.0.i, %spirv.JointMatrixINTEL._float_8_16_3_3_2 addrspace(4)* noundef %sub_c.sroa.0.0.i, i32 noundef 3) #3
  %add.i = add nuw nsw i32 %k.0.i, 16
  br label %for.cond.i

for.body59.i:                                     ; preds = %for.cond52.i
  %call.i236.i = tail call spir_func noundef float @_Z28__spirv_VectorExtractDynamicIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm16ELm16ELN5__spv9MatrixUseE1ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EET_PNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEm(%spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* noundef %sub_b.sroa.0.0.i, i64 noundef %conv53.i) #3
  %call.i171.i = tail call spir_func noundef float @_Z27__spirv_ConvertFToTF32INTELf(float noundef %call.i236.i) #3
  %call.i243.i = tail call spir_func noundef %spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* @_Z27__spirv_VectorInsertDynamicIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm16ELm16ELN5__spv9MatrixUseE1ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEPNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEESG_T_m(%spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* noundef %sub_b.sroa.0.0.i, float noundef %call.i171.i, i64 noundef %conv53.i) #3
  %inc74.i = add nuw nsw i32 %i51.0.i, 1
  br label %for.cond52.i

for.cond82.i:                                     ; preds = %for.cond.i, %for.body87.i
  %sub_a.sroa.0.2.i = phi %spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* [ %call5.i.i, %for.body87.i ], [ %sub_a.sroa.0.0.i, %for.cond.i ]
  %i81.0.i = phi i32 [ %inc96.i, %for.body87.i ], [ 0, %for.cond.i ]
  %conv83.i = zext i32 %i81.0.i to i64
  %call.i247.i = tail call spir_func noundef i64 @_Z38__spirv_JointMatrixWorkItemLengthINTELIN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEmPNS8_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT3_EXT4_EXT2_EEE(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef %sub_a.sroa.0.2.i) #3
  %cmp85.i = icmp ugt i64 %call.i247.i, %conv83.i
  br i1 %cmp85.i, label %for.body87.i, label %_ZZZ15matrix_multiplyIffLm16ELm32ELm32ELm32ELm16ELm32EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN4sycl3_V17handlerEE_clESC_ENKUlNSA_7nd_itemILi2EEEE_clESF_.exit

for.body87.i:                                     ; preds = %for.cond82.i
  %call.i269.i = tail call spir_func noundef float @_Z28__spirv_VectorExtractDynamicIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EET_PNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEm(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef %sub_a.sroa.0.2.i, i64 noundef %conv83.i) #3
  %call.i276.i = tail call spir_func noundef float @_Z28__spirv_VectorExtractDynamicIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EET_PNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEm(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef %sub_a.sroa.0.2.i, i64 noundef %conv83.i) #3
  %mul.i.i = fmul float %call.i276.i, 2.000000e+00
  %call5.i.i = tail call spir_func noundef %spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* @_Z27__spirv_VectorInsertDynamicIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEPNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEESG_T_m(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef %sub_a.sroa.0.2.i, float noundef %mul.i.i, i64 noundef %conv83.i) #3
  %inc96.i = add nuw nsw i32 %i81.0.i, 1
  br label %for.cond82.i

_ZZZ15matrix_multiplyIffLm16ELm32ELm32ELm32ELm16ELm32EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN4sycl3_V17handlerEE_clESC_ENKUlNSA_7nd_itemILi2EEEE_clESF_.exit: ; preds = %for.cond82.i
  tail call spir_func void @_Z29__spirv_JointMatrixStoreINTELIffLm8ELm16ELN5__spv9MatrixUseE2ELNS0_12MatrixLayoutE3ELNS0_5Scope4FlagE3EEvPT_PNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEmS2_S4_i(float addrspace(4)* noundef %call.ascast.i.i, %spirv.JointMatrixINTEL._float_8_16_3_3_2 addrspace(4)* noundef %sub_c.sroa.0.0.i, i64 noundef %_arg_N, i32 noundef 0, i32 noundef 3, i32 noundef 0) #3
  call void @__itt_offload_wi_finish_wrapper()
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent
declare dso_local spir_func noundef %spirv.JointMatrixINTEL._float_8_16_3_3_2 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIffLm8ELm16ELN5__spv9MatrixUseE2ELNS0_12MatrixLayoutE3ELNS0_5Scope4FlagE3EEPNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS2_S4_i(float addrspace(4)* noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef %spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEPNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mSA_SC_i(float addrspace(4)* noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef %spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* @_Z28__spirv_JointMatrixLoadINTELIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm16ELm16ELN5__spv9MatrixUseE1ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEPNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mSA_SC_i(float addrspace(4)* noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef i64 @_Z38__spirv_JointMatrixWorkItemLengthINTELIN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEmPNS8_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT3_EXT4_EXT2_EEE(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef float @_Z27__spirv_ConvertFToTF32INTELf(float noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef float @_Z28__spirv_VectorExtractDynamicIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EET_PNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEm(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef %spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* @_Z27__spirv_VectorInsertDynamicIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm16ELN5__spv9MatrixUseE0ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEPNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEESG_T_m(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef, float noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef i64 @_Z38__spirv_JointMatrixWorkItemLengthINTELIN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm16ELm16ELN5__spv9MatrixUseE1ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEmPNS8_24__spirv_JointMatrixINTELIT_XT0_EXT1_EXT3_EXT4_EXT2_EEE(%spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef float @_Z28__spirv_VectorExtractDynamicIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm16ELm16ELN5__spv9MatrixUseE1ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EET_PNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEm(%spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef %spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* @_Z27__spirv_VectorInsertDynamicIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm16ELm16ELN5__spv9MatrixUseE1ELNS8_12MatrixLayoutE0ELNS8_5Scope4FlagE3EEPNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEESG_T_m(%spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* noundef, float noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef %spirv.JointMatrixINTEL._float_8_16_3_3_2 addrspace(4)* @_Z27__spirv_JointMatrixMadINTELIN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32EfLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS9_1ELS9_2ELNS8_12MatrixLayoutE0ELSA_0ELSA_3ELNS8_5Scope4FlagE3EEPNS8_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT9_EXT10_EXT6_EEEPNSD_IT_XT1_EXT2_EXT7_EXT10_EXT4_EEEPNSD_ISH_XT2_EXT3_EXT8_EXT10_EXT5_EEESG_SC_(%spirv.JointMatrixINTEL._tf32_8_16_0_3_0 addrspace(4)* noundef, %spirv.JointMatrixINTEL._tf32_16_16_0_3_1 addrspace(4)* noundef, %spirv.JointMatrixINTEL._float_8_16_3_3_2 addrspace(4)* noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func void @_Z29__spirv_JointMatrixStoreINTELIffLm8ELm16ELN5__spv9MatrixUseE2ELNS0_12MatrixLayoutE3ELNS0_5Scope4FlagE3EEvPT_PNS0_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEmS2_S4_i(float addrspace(4)* noundef, %spirv.JointMatrixINTEL._float_8_16_3_3_2 addrspace(4)* noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

declare void @__itt_offload_wi_finish_wrapper()

attributes #0 = { convergent norecurse "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="matrix-tf32-test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent }
