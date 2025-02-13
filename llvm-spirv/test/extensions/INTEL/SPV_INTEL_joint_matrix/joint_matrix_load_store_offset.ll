; This is an adapted copy of test/extensions/KHR/SPV_KHR_cooperative_matrix/cooperative_matrix.ll

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability CooperativeMatrixKHR
; CHECK-SPIRV-DAG: Capability CooperativeMatrixOffsetInstructionsINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_joint_matrix"
; CHECK-SPIRV-DAG: TypeInt [[#Int16Ty:]] 16 0
; CHECK-SPIRV-DAG: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[#Int64Ty:]] 64 0
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const0:]] 0
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const1:]] 1
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const2:]] 2
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const3:]] 3
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const16:]] 16
; CHECK-SPIRV-DAG: Constant [[#Int64Ty]] [[#Const128:]] 128 0
; CHECK-SPIRV-DAG: Constant [[#Int64Ty:]] [[#Const256:]] 256 0
; CHECK-SPIRV-DAG: TypeFloat [[#Float32Ty:]] 32
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#MatTy1:]] [[#Float32Ty]] [[#Const3]] [[#Const1]] [[#Const16]] [[#Const2]]
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#MatTy2:]] [[#Int16Ty]] [[#Const3]] [[#Const1]] [[#Const16]] [[#Const0:]]
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#MatTy3:]] [[#Int16Ty]] [[#Const3]] [[#Const16]] [[#Const16]] [[#Const1]]
; CHECK-SPIRV: CooperativeMatrixLoadOffsetINTEL [[#MatTy1]] [[#]] [[#Ptr1:]] [[#]] [[#Index1:]] [[#Const0]] [[#Const128]] 0
; CHECK-SPIRV: CooperativeMatrixLoadOffsetINTEL [[#MatTy2]] [[#Load2:]] [[#]] [[#Index2:]] [[#]] [[#Const0]] [[#Const128]] 0
; CHECK-SPIRV: CooperativeMatrixLoadOffsetINTEL [[#MatTy3]] [[#Load3:]] [[#]] [[#]] [[#]] [[#Const2:]] [[#Const256:]] 0
; CHECK-SPIRV: CooperativeMatrixMulAddKHR [[#MatTy1]] [[#]] [[#Load2]] [[#Load3]] [[#Result:]] 64
; CHECK-SPIRV: CooperativeMatrixStoreOffsetINTEL [[#Ptr1]] [[#Index2]] [[#Index1]] [[#Result]] [[#Const0]] [[#Const128]] 0 

; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) @_Z94__spirv_CooperativeMatrixLoadOffsetINTEL_RPU3AS144__spirv_CooperativeMatrixKHR__float_3_1_16_2PU3AS1fiiili(ptr addrspace(1) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0, i64 128, i32 0) 
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", i16, 3, 1, 16, 0) @"_Z94__spirv_CooperativeMatrixLoadOffsetINTEL_RPU3AS144__spirv_CooperativeMatrixKHR__short_3_1_16_0PU3AS138class.sycl::_V1::ext::oneapi::bfloat16iiili"(ptr addrspace(1) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0, i64 128, i32 0)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) @"_Z95__spirv_CooperativeMatrixLoadOffsetINTEL_RPU3AS145__spirv_CooperativeMatrixKHR__short_3_16_16_1PU3AS138class.sycl::_V1::ext::oneapi::bfloat16iiili"(ptr addrspace(1) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 2, i64 256, i32 0)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) @_Z34__spirv_CooperativeMatrixMulAddKHRPU3AS144__spirv_CooperativeMatrixKHR__short_3_1_16_0PU3AS145__spirv_CooperativeMatrixKHR__short_3_16_16_1PU3AS144__spirv_CooperativeMatrixKHR__float_3_1_16_2i(target("spirv.CooperativeMatrixKHR", i16, 3, 1, 16, 0) %{{.*}}, target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) %{{.*}}, target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) %{{.*}}, i32 64)
; CHECK-LLVM: call spir_func void @_Z41__spirv_CooperativeMatrixStoreOffsetINTELPU3AS1fiiPU3AS144__spirv_CooperativeMatrixKHR__float_3_1_16_2ili(ptr addrspace(1) %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) %{{.*}}, i32 0, i64 128, i32 0)

; ModuleID = 'joint_matrix_all_sizes.cpp'
source_filename = "joint_matrix_all_sizes.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }

$_ZTSZZ15matrix_multiply = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiply(ptr addrspace(1) noundef align 4 %_arg_accC, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_accC2, ptr noundef byval(%"class.sycl::_V1::id") align 8 %_arg_accC3, i64 noundef %_arg_sg_size, ptr addrspace(1) noundef readonly align 2 %_arg_accA, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_accA5, ptr noundef byval(%"class.sycl::_V1::id") align 8 %_arg_accA6, ptr addrspace(1) noundef readonly align 2 %_arg_accB, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_accB8, ptr noundef byval(%"class.sycl::_V1::id") align 8 %_arg_accB9) comdat {
entry:
  %agg.tmp11.sroa.0.sroa.2.0._arg_accC2.ascast.sroa_idx = getelementptr inbounds i8, ptr %_arg_accC2, i64 8
  %agg.tmp11.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp11.sroa.0.sroa.2.0._arg_accC2.ascast.sroa_idx, align 8
  %agg.tmp12.sroa.0.sroa.0.0.copyload = load i64, ptr %_arg_accC3, align 8
  %agg.tmp12.sroa.0.sroa.2.0._arg_accC3.ascast.sroa_idx = getelementptr inbounds i8, ptr %_arg_accC3, i64 8
  %agg.tmp12.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp12.sroa.0.sroa.2.0._arg_accC3.ascast.sroa_idx, align 8
  %mul.i6.i.i.i.i = mul i64 %agg.tmp12.sroa.0.sroa.0.0.copyload, %agg.tmp11.sroa.0.sroa.2.0.copyload
  %0 = getelementptr float, ptr addrspace(1) %_arg_accC, i64 %mul.i6.i.i.i.i
  %add.ptr.i = getelementptr float, ptr addrspace(1) %0, i64 %agg.tmp12.sroa.0.sroa.2.0.copyload
  %agg.tmp15.sroa.0.sroa.2.0._arg_accA5.ascast.sroa_idx = getelementptr inbounds i8, ptr %_arg_accA5, i64 8
  %agg.tmp15.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp15.sroa.0.sroa.2.0._arg_accA5.ascast.sroa_idx, align 8
  %agg.tmp16.sroa.0.sroa.0.0.copyload = load i64, ptr %_arg_accA6, align 8
  %agg.tmp16.sroa.0.sroa.2.0._arg_accA6.ascast.sroa_idx = getelementptr inbounds i8, ptr %_arg_accA6, i64 8
  %agg.tmp16.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp16.sroa.0.sroa.2.0._arg_accA6.ascast.sroa_idx, align 8
  %mul.i6.i.i.i.i91 = mul i64 %agg.tmp16.sroa.0.sroa.0.0.copyload, %agg.tmp15.sroa.0.sroa.2.0.copyload
  %1 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %_arg_accA, i64 %mul.i6.i.i.i.i91
  %add.ptr.i92 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %1, i64 %agg.tmp16.sroa.0.sroa.2.0.copyload
  %agg.tmp19.sroa.0.sroa.2.0._arg_accB8.ascast.sroa_idx = getelementptr inbounds i8, ptr %_arg_accB8, i64 8
  %agg.tmp19.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp19.sroa.0.sroa.2.0._arg_accB8.ascast.sroa_idx, align 8
  %agg.tmp20.sroa.0.sroa.0.0.copyload = load i64, ptr %_arg_accB9, align 8
  %agg.tmp20.sroa.0.sroa.2.0._arg_accB9.ascast.sroa_idx = getelementptr inbounds i8, ptr %_arg_accB9, i64 8
  %agg.tmp20.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp20.sroa.0.sroa.2.0._arg_accB9.ascast.sroa_idx, align 8
  %mul.i6.i.i.i.i107 = mul i64 %agg.tmp20.sroa.0.sroa.0.0.copyload, %agg.tmp19.sroa.0.sroa.2.0.copyload
  %2 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %_arg_accB, i64 %mul.i6.i.i.i.i107
  %add.ptr.i108 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %2, i64 %agg.tmp20.sroa.0.sroa.2.0.copyload
  %3 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, i64 8), align 8
  %cmp.i28 = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i28)
  %4 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %cmp.i24 = icmp ult i64 %4, 2147483648
  tail call void @llvm.assume(i1 %cmp.i24)
  %5 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, i64 8), align 8
  %cmp.i35 = icmp ult i64 %5, 2147483648
  tail call void @llvm.assume(i1 %cmp.i35)
  %sub.i = sub nsw i64 %3, %5
  %6 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32
  %cmp.i31 = icmp ult i64 %6, 2147483648
  tail call void @llvm.assume(i1 %cmp.i31)
  %sub5.i = sub nsw i64 %4, %6
  %add.i7.i.i.i.i.i = add i64 %mul.i6.i.i.i.i, %agg.tmp12.sroa.0.sroa.2.0.copyload
  %idx.neg.i.i = sub i64 0, %add.i7.i.i.i.i.i
  %add.ptr.i.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i, i64 %idx.neg.i.i
  %div.i = udiv i64 %sub5.i, %_arg_sg_size
  %conv.i = trunc i64 %sub.i to i32
  %div.i.tr = trunc i64 %div.i to i32
  %conv2.i = shl i32 %div.i.tr, 4
  %call4.i = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) @_Z40__spirv_CooperativeMatrixLoadOffsetINTELIU3AS1ffLm1ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_iiS3_mi(ptr addrspace(1) noundef %add.ptr.i.i, i32 noundef %conv.i, i32 noundef %conv2.i, i32 noundef 0, i64 noundef 128, i32 noundef 0)
  %add.i7.i.i.i.i.i118 = add i64 %mul.i6.i.i.i.i91, %agg.tmp16.sroa.0.sroa.2.0.copyload
  %idx.neg.i.i119 = sub i64 0, %add.i7.i.i.i.i.i118
  %add.ptr.i.i120 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %add.ptr.i92, i64 %idx.neg.i.i119
  %add.i7.i.i.i.i.i126 = add i64 %mul.i6.i.i.i.i107, %agg.tmp20.sroa.0.sroa.2.0.copyload
  %idx.neg.i.i127 = sub i64 0, %add.i7.i.i.i.i.i126
  %add.ptr.i.i128 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %add.ptr.i108, i64 %idx.neg.i.i127
  %conv2.i60 = shl i32 %div.i.tr, 5
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.body.i, %entry
  %sub_c.i.sroa.0.0 = phi target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) [ %call4.i, %entry ], [ %call.i63, %for.body.i ]
  %k.0.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  %cmp.i = icmp samesign ult i32 %k.0.i, 8
  br i1 %cmp.i, label %for.body.i, label %_ZZZ15matrix_multiplyIfN4sycl3_V13ext6oneapi8bfloat16ELm16ELm128ELm128ELi2ELm1ELm16ELm16E4multIS4_Lm1ELm16ELm16EEEvR10big_matrixIT_XT1_EXT2_EERS7_IT0_XT1_EXT3_EERS7_ISB_XdvT3_T4_EXmlT2_T4_EEENKUlRNS1_7handlerEE_clESH_ENKUlNS1_7nd_itemILi2EEEE_clESK_.exit

for.body.i:                                       ; preds = %for.cond.i
  %7 = shl nuw nsw i32 %k.0.i, 4
  %call3.i50 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 1, 16, 0) @_Z40__spirv_CooperativeMatrixLoadOffsetINTELIU3AS1KN4sycl3_V13ext6oneapi8bfloat16ES4_Lm1ELm16ELN5__spv9MatrixUseE0ELNS6_12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_iiS8_mi(ptr addrspace(1) noundef %add.ptr.i.i120, i32 noundef %conv.i, i32 noundef %7, i32 noundef 0, i64 noundef 128, i32 noundef 0)
  %8 = shl nuw nsw i32 %k.0.i, 3
  %call3.i61 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) @_Z40__spirv_CooperativeMatrixLoadOffsetINTELIU3AS1KN4sycl3_V13ext6oneapi8bfloat16ES4_Lm16ELm16ELN5__spv9MatrixUseE1ELNS6_12MatrixLayoutE2ELNS6_5Scope4FlagE3EEPNS6_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_iiS8_mi(ptr addrspace(1) noundef %add.ptr.i.i128, i32 noundef %8, i32 noundef %conv2.i60, i32 noundef 2, i64 noundef 256, i32 noundef 0)
  %call.i63 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) @_Z34__spirv_CooperativeMatrixMulAddKHRIN4sycl3_V13ext6oneapi8bfloat16ES4_fLm1ELm16ELm16ELN5__spv9MatrixUseE0ELS6_1ELS6_2ELNS5_12MatrixLayoutE0ELS7_0ELS7_0ELNS5_5Scope4FlagE3EEPNS5_28__spirv_CooperativeMatrixKHRIT1_XT11_EXT2_EXT4_EXT7_EEEPNSA_IT_XT11_EXT2_EXT3_EXT5_EEEPNSA_IT0_XT11_EXT3_EXT4_EXT6_EEESD_m(target("spirv.CooperativeMatrixKHR", i16, 3, 1, 16, 0) noundef %call3.i50, target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) noundef %call3.i61, target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) noundef %sub_c.i.sroa.0.0, i64 noundef 64)
  %add.i = add nuw nsw i32 %k.0.i, 1
  br label %for.cond.i

_ZZZ15matrix_multiplyIfN4sycl3_V13ext6oneapi8bfloat16ELm16ELm128ELm128ELi2ELm1ELm16ELm16E4multIS4_Lm1ELm16ELm16EEEvR10big_matrixIT_XT1_EXT2_EERS7_IT0_XT1_EXT3_EERS7_ISB_XdvT3_T4_EXmlT2_T4_EEENKUlRNS1_7handlerEE_clESH_ENKUlNS1_7nd_itemILi2EEEE_clESK_.exit: ; preds = %for.cond.i
  tail call spir_func void @_Z41__spirv_CooperativeMatrixStoreOffsetINTELIU3AS1ffLm1ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEvPT_iiPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEES3_mi(ptr addrspace(1) noundef %add.ptr.i.i, i32 noundef %conv.i, i32 noundef %conv2.i, target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) noundef %sub_c.i.sroa.0.0, i32 noundef 0, i64 noundef 128, i32 noundef 0)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef)

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) @_Z40__spirv_CooperativeMatrixLoadOffsetINTELIU3AS1ffLm1ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_iiS3_mi(ptr addrspace(1) noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i32 noundef)

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 1, 16, 0) @_Z40__spirv_CooperativeMatrixLoadOffsetINTELIU3AS1KN4sycl3_V13ext6oneapi8bfloat16ES4_Lm1ELm16ELN5__spv9MatrixUseE0ELNS6_12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_iiS8_mi(ptr addrspace(1) noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i32 noundef)

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) @_Z40__spirv_CooperativeMatrixLoadOffsetINTELIU3AS1KN4sycl3_V13ext6oneapi8bfloat16ES4_Lm16ELm16ELN5__spv9MatrixUseE1ELNS6_12MatrixLayoutE2ELNS6_5Scope4FlagE3EEPNS6_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEEPT_iiS8_mi(ptr addrspace(1) noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i32 noundef)

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) @_Z34__spirv_CooperativeMatrixMulAddKHRIN4sycl3_V13ext6oneapi8bfloat16ES4_fLm1ELm16ELm16ELN5__spv9MatrixUseE0ELS6_1ELS6_2ELNS5_12MatrixLayoutE0ELS7_0ELS7_0ELNS5_5Scope4FlagE3EEPNS5_28__spirv_CooperativeMatrixKHRIT1_XT11_EXT2_EXT4_EXT7_EEEPNSA_IT_XT11_EXT2_EXT3_EXT5_EEEPNSA_IT0_XT11_EXT3_EXT4_EXT6_EEESD_m(target("spirv.CooperativeMatrixKHR", i16, 3, 1, 16, 0) noundef, target("spirv.CooperativeMatrixKHR", i16, 3, 16, 16, 1) noundef, target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) noundef, i64 noundef)

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z41__spirv_CooperativeMatrixStoreOffsetINTELIU3AS1ffLm1ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEvPT_iiPNS1_28__spirv_CooperativeMatrixKHRIT0_XT5_EXT1_EXT2_EXT3_EEES3_mi(ptr addrspace(1) noundef, i32 noundef, i32 noundef, target("spirv.CooperativeMatrixKHR", float, 3, 1, 16, 2) noundef, i32 noundef, i64 noundef, i32 noundef)
