; This is an adapted copy of test/extensions/KHR/SPV_KHR_cooperative_matrix/cooperative_matrix.ll

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability CooperativeMatrixKHR
; CHECK-SPIRV-DAG: Capability CooperativeMatrixCheckedInstructionsINTEL
; CHECK-SPIRV-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_joint_matrix"
; CHECK-SPIRV-DAG: TypeInt [[#Int8Ty:]] 8 0
; CHECK-SPIRV-DAG: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const12:]] 12
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const48:]] 48
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const0:]] 0
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const3:]] 3
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const2:]] 2
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const1:]] 1
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#MatTy1:]] [[#Int32Ty]] [[#Const3]] [[#Const12]] [[#Const12]] [[#Const2]]
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#MatTy2:]] [[#Int8Ty]] [[#Const3]] [[#Const12]] [[#Const48]] [[#Const0]]
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#MatTy3:]] [[#Int8Ty]] [[#Const2]] [[#Const48]] [[#Const12]] [[#Const1]]
; CHECK-SPIRV: CooperativeMatrixConstructCheckedINTEL [[#MatTy1]]
; CHECK-SPIRV: CooperativeMatrixLoadCheckedINTEL [[#MatTy2]] [[#Load1:]]
; TODO: Pass Matrix Type Id instead of Matrix Id to CooperativeMatrixLengthKHR.
; CHECK-SPIRV: CooperativeMatrixLengthKHR [[#Int32Ty]] [[#]] [[#Load1]]
; CHECK-SPIRV: CooperativeMatrixLoadCheckedINTEL [[#MatTy3]]
; CHECK-SPIRV: CooperativeMatrixMulAddKHR [[#MatTy1]]
; CHECK-SPIRV: CooperativeMatrixStoreCheckedINTEL

; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z46__spirv_CooperativeMatrixConstructCheckedINTELiiiii(i32 4, i32 4, i32 12, i32 12, i32 %_arg_Initvalue)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) @_Z95__spirv_CooperativeMatrixLoadCheckedINTEL_RPU3AS144__spirv_CooperativeMatrixKHR__char_3_12_48_0PU3AS4ciiiiili(ptr addrspace(4) %[[MatrixPtr:[%0-9a-z.]+]], i32 0, i32 0, i32 0, i32 12, i32 48, i64 %_arg_K, i32 1)
; CHECK-LLVM: call spir_func i32 @_Z34__spirv_CooperativeMatrixLengthKHRPU3AS144__spirv_CooperativeMatrixKHR__char_3_12_48_0(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) @_Z95__spirv_CooperativeMatrixLoadCheckedINTEL_RPU3AS144__spirv_CooperativeMatrixKHR__char_2_48_12_1PU3AS4ciiiiil
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHRPU3AS144__spirv_CooperativeMatrixKHR__char_3_12_48_0PU3AS144__spirv_CooperativeMatrixKHR__char_2_48_12_1PU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2i(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) %{{.*}}, target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) %{{.*}}, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2)
; CHECK-LLVM: call spir_func void @_Z42__spirv_CooperativeMatrixStoreCheckedINTELPU3AS4iiiPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_2iiili(ptr addrspace(4) %{{.*}}, i32 0, i32 0, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2)

; ModuleID = 'test-matrix-opaque.bc'
source_filename = "matrix-int8-test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }

$_ZTSZZ15matrix_multiply = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiply(ptr addrspace(1) noundef align 1 %_arg_accA, ptr addrspace(1) noundef align 1 %_arg_accB, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_accB5, ptr noundef byval(%"class.sycl::_V1::id") align 8 %_arg_accB6, ptr addrspace(1) noundef align 4 %_arg_accC, i64 noundef %_arg_N, i64 noundef %_arg_K, i32 noundef %_arg_Initvalue) local_unnamed_addr #0 comdat {
entry:
  %sub_c.sroa.0.i = alloca target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), align 8
  %ref.tmp29.sroa.0.i = alloca target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), align 8
  %agg.tmp15.sroa.0.sroa.2.0..sroa_idx = getelementptr inbounds %"class.sycl::_V1::range", ptr %_arg_accB5, i64 0, i32 0, i32 0, i64 1
  %agg.tmp15.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp15.sroa.0.sroa.2.0..sroa_idx, align 8
  %agg.tmp16.sroa.0.sroa.0.0.copyload = load i64, ptr %_arg_accB6, align 8
  %agg.tmp16.sroa.0.sroa.2.0..sroa_idx = getelementptr inbounds %"class.sycl::_V1::id", ptr %_arg_accB6, i64 0, i32 0, i32 0, i64 1
  %agg.tmp16.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp16.sroa.0.sroa.2.0..sroa_idx, align 8
  %mul.i4.i.i.i.i45 = mul i64 %agg.tmp16.sroa.0.sroa.0.0.copyload, %agg.tmp15.sroa.0.sroa.2.0.copyload
  %add.i6.i.i.i.i46 = add i64 %mul.i4.i.i.i.i45, %agg.tmp16.sroa.0.sroa.2.0.copyload
  %add.ptr.i47 = getelementptr inbounds i8, ptr addrspace(1) %_arg_accB, i64 %add.i6.i.i.i.i46
  %0 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %1 = extractelement <3 x i64> %0, i64 1
  %2 = extractelement <3 x i64> %0, i64 0
  %3 = load <3 x i64>, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32
  %4 = extractelement <3 x i64> %3, i64 1
  %5 = extractelement <3 x i64> %3, i64 0
  %cmp.i.i = icmp ult i64 %1, 2147483648
  %cmp.i54.i = icmp ult i64 %2, 2147483648
  %cmp.i56.i = icmp ult i64 %4, 2147483648
  %sub.i = sub nsw i64 %1, %4
  %cmp.i58.i = icmp ult i64 %5, 2147483648
  %sub5.i = sub nsw i64 %2, %5
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %sub_c.sroa.0.i)
  %call.i.i = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z46__spirv_CooperativeMatrixConstructCheckedINTEL(i32 noundef 4, i32 noundef 4, i32 noundef 12, i32 noundef 12, i32 noundef %_arg_Initvalue) #4
  store target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %call.i.i, ptr %sub_c.sroa.0.i, align 8
  %mul.i = mul nsw i64 %sub.i, 12
  %div2452.i = lshr i64 %sub5.i, 4
  %mul26.i = mul i64 %div2452.i, 48
  %div.i = udiv i64 %_arg_K, 48
  %mul11.i = mul i64 %mul.i, %_arg_K
  %add.ptr.i93.i = getelementptr inbounds i8, ptr addrspace(1) %_arg_accA, i64 %mul11.i
  %idx.neg.i.i104.i = sub i64 0, %add.i6.i.i.i.i46
  %add.ptr.i.i105141.i = getelementptr i8, ptr addrspace(1) %add.ptr.i47, i64 %mul26.i
  %mul22.i = shl i64 %_arg_N, 2
  %add.ptr.i108140.i = getelementptr i8, ptr addrspace(1) %add.ptr.i.i105141.i, i64 %idx.neg.i.i104.i
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.body.i, %entry
  %k.0.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  %conv.i = zext i32 %k.0.i to i64
  %cmp.i = icmp ugt i64 %div.i, %conv.i
  br i1 %cmp.i, label %for.body.i, label %_ZZZ15matrix_multiplyIiaLm24ELm96ELm24ELm96ELm24ELm24EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN4sycl3_V17handlerEE_clESC_ENKUlNSA_7nd_itemILi2EEEE_clESF_.exit

for.body.i:                                       ; preds = %for.cond.i
  %mul12.i = mul nsw i32 %k.0.i, 48
  %conv13.i = zext i32 %mul12.i to i64
  %add.ptr.i96.i = getelementptr inbounds i8, ptr addrspace(1) %add.ptr.i93.i, i64 %conv13.i
  %call.ascast.i66.i = addrspacecast ptr addrspace(1) %add.ptr.i96.i to ptr addrspace(4)
  %call1.i.i = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_1(ptr addrspace(4) noundef %call.ascast.i66.i, i32 noundef 0, i32 noundef 0, i32 noundef 0, i32 noundef 12, i32 noundef 48, i64 noundef %_arg_K, i32 noundef 1) #4
  %len = tail call spir_func noundef i32 @_Z34__spirv_CooperativeMatrixLengthKHR(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) %call1.i.i)
  %div20.i = mul nsw i32 %k.0.i, 12
  %conv21.i = zext i32 %div20.i to i64
  %mul23.i = mul i64 %mul22.i, %conv21.i
  %add.ptr.i111.i = getelementptr i8, ptr addrspace(1) %add.ptr.i108140.i, i64 %mul23.i
  %call.ascast.i72.i = addrspacecast ptr addrspace(1) %add.ptr.i111.i to ptr addrspace(4)
  %call1.i73.i = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_2(ptr addrspace(4) noundef %call.ascast.i72.i, i32 noundef 0, i32 noundef 0, i32 noundef 0, i32 noundef 48, i32 noundef 12, i64 noundef %mul22.i) #4
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %ref.tmp29.sroa.0.i)
  %sub_c.sroa.0.i.0.sub_c.sroa.0.i.0.sub_c.sroa.0.0.sub_c.sroa.0.0.sub_c.sroa.0.0.125.i = load target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), ptr %sub_c.sroa.0.i, align 8
  %call.i77.i = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) noundef %call1.i.i, target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) noundef %call1.i73.i, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef %sub_c.sroa.0.i.0.sub_c.sroa.0.i.0.sub_c.sroa.0.0.sub_c.sroa.0.0.sub_c.sroa.0.0.125.i, i32 noundef 12) #4
  store target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %call.i77.i, ptr %ref.tmp29.sroa.0.i, align 8
  %ref.tmp29.sroa.0.i.0.ref.tmp29.sroa.0.i.0.ref.tmp29.sroa.0.0.ref.tmp29.sroa.0.0.ref.tmp29.sroa.0.0..i = load i64, ptr %ref.tmp29.sroa.0.i, align 8
  store i64 %ref.tmp29.sroa.0.i.0.ref.tmp29.sroa.0.i.0.ref.tmp29.sroa.0.0.ref.tmp29.sroa.0.0.ref.tmp29.sroa.0.0..i, ptr %sub_c.sroa.0.i, align 8
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %ref.tmp29.sroa.0.i)
  %add.i = add nuw nsw i32 %k.0.i, 1
  br label %for.cond.i

_ZZZ15matrix_multiplyIiaLm24ELm96ELm24ELm96ELm24ELm24EEvR10big_matrixIT_XT5_EXT6_EERS0_IT0_XT1_EXT2_EERS0_IS4_XT3_EXT4_EEENKUlRN4sycl3_V17handlerEE_clESC_ENKUlNSA_7nd_itemILi2EEEE_clESF_.exit: ; preds = %for.cond.i
  %mul37.i = mul i64 %mul.i, %_arg_N
  %add.ptr.i.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_accC, i64 %mul37.i
  %mul39.i = mul nuw i64 %div2452.i, 12
  %add.ptr.i81.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i.i, i64 %mul39.i
  %call.ascast.i.i = addrspacecast ptr addrspace(1) %add.ptr.i81.i to ptr addrspace(4)
  %sub_c.sroa.0.i.0.sub_c.sroa.0.i.0.sub_c.sroa.0.0.sub_c.sroa.0.0.sub_c.sroa.0.0..i = load target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), ptr %sub_c.sroa.0.i, align 8
  tail call spir_func void @_Z42__spirv_CooperativeMatrixStoreCheckedINTEL(ptr addrspace(4) noundef %call.ascast.i.i, i32 noundef 0, i32 noundef 0, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef %sub_c.sroa.0.i.0.sub_c.sroa.0.i.0.sub_c.sroa.0.0.sub_c.sroa.0.0.sub_c.sroa.0.0..i, i32 noundef 0, i32 noundef 12, i32 noundef 12, i64 noundef %_arg_N, i32 noundef 1) #4
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %sub_c.sroa.0.i)
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z46__spirv_CooperativeMatrixConstructCheckedINTEL(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

declare dso_local spir_func noundef i32 @_Z34__spirv_CooperativeMatrixLengthKHR(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) noundef)

; Function Attrs: convergent
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_1(ptr addrspace(4) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) @_Z41__spirv_CooperativeMatrixLoadCheckedINTEL_2(ptr addrspace(4) noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i8, 3, 12, 48, 0) noundef, target("spirv.CooperativeMatrixKHR", i8, 2, 48, 12, 1) noundef, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func void @_Z42__spirv_CooperativeMatrixStoreCheckedINTEL(ptr addrspace(4) noundef, i32 noundef, i32 noundef, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) noundef, i32 noundef, i32 noundef, i32 noundef, i64 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #3

attributes #0 = { convergent norecurse "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="matrix-int8-test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { convergent }
