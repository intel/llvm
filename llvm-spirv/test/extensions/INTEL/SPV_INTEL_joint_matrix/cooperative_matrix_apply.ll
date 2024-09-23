;; compiled from joint_matrix_apply_bf16.cpp from intel/llvm with some modifications

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability CooperativeMatrixKHR
; CHECK-SPIRV-DAG: Capability CooperativeMatrixInvocationInstructionsINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_joint_matrix"
; CHECK-SPIRV-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#MatTy:]]
; CHECK-SPIRV: CompositeConstruct [[#MatTy]] [[#Mat:]]
; CHECK-SPIRV: PtrCastToGeneric [[#]] [[#Ptr:]] [[#]]
; CHECK-SPIRV: CooperativeMatrixApplyFunctionINTEL [[#MatTy]] [[#Apply:]] [[#Ptr]] [[#Mat]]
; CHECK-SPIRV: CooperativeMatrixStoreKHR [[#]] [[#Apply]]

; CHECK-LLVM: %[[Mat:[%0-9a-z.]+]] = call spir_func target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) @"_Z26__spirv_CompositeConstructP38class.sycl::_V1::ext::oneapi::bfloat16"
; CHECK-LLVM: %[[Apply:[%0-9a-z.]+]] = call spir_func target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) @"_Z43__spirv_CooperativeMatrixApplyFunctionINTELPU3AS477class.sycl::_V1::ext::oneapi::experimental::matrix::helper::reference_wrapperPU3AS144__spirv_CooperativeMatrixKHR__short_3_8_16_0"(ptr addrspace(4) %ref.tmp.ascast.i21, target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) %[[Mat]])
; CHECK-LLVM: call spir_func void @"_Z33__spirv_CooperativeMatrixStoreKHRPU3AS138class.sycl::_V1::ext::oneapi::bfloat16PU3AS144__spirv_CooperativeMatrixKHR__short_3_8_16_0il"(ptr addrspace(1) %{{.*}}, target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) %[[Apply]], i32 0, i64 0)

; ModuleID = 'matrix_apply.bc'
source_filename = "../llvm/sycl/test-e2e/Matrix/joint_matrix_apply_bf16.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [2 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::ext::oneapi::experimental::matrix::helper::reference_wrapper" = type { ptr addrspace(4) }
%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%class.anon.0 = type <{ %"class.sycl::_V1::accessor", %class.anon, [7 x i8] }>
%"class.sycl::_V1::accessor" = type { %"class.sycl::_V1::detail::AccessorImplDevice", %union.anon }
%"class.sycl::_V1::detail::AccessorImplDevice" = type { %"class.sycl::_V1::id", %"class.sycl::_V1::range", %"class.sycl::_V1::range" }
%union.anon = type { ptr addrspace(1) }
%class.anon = type { i8 }

$_ZTSZZ17matrix_verify_addIN4sycl3_V13ext6oneapi8bfloat16ELm16ELm32EZ4mainEUlRS4_E_EvNS1_5queueER10big_matrixIT_XT0_EXT1_EERNS1_8nd_rangeILi2EEEfOT2_ENKUlRNS1_7handlerEE_clESI_EUlNS1_7nd_itemILi2EEEE_ = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ17matrix_verify_addIN4sycl3_V13ext6oneapi8bfloat16ELm16ELm32EZ4mainEUlRS4_E_EvNS1_5queueER10big_matrixIT_XT0_EXT1_EERNS1_8nd_rangeILi2EEEfOT2_ENKUlRNS1_7handlerEE_clESI_EUlNS1_7nd_itemILi2EEEE_(ptr addrspace(1) noundef align 2 %_arg_accA, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_accA1, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_accA2, ptr noundef byval(%"class.sycl::_V1::id") align 8 %_arg_accA3) local_unnamed_addr {
entry:
  %ref.tmp.i20 = alloca %"class.sycl::_V1::ext::oneapi::experimental::matrix::helper::reference_wrapper", align 8
  %agg.tmp.i17 = alloca %"class.sycl::_V1::ext::oneapi::bfloat16", align 2
  %ref.tmp6.i = alloca float, align 4
  %__SYCLKernel = alloca %class.anon.0, align 8
  %__SYCLKernel.ascast = addrspacecast ptr %__SYCLKernel to ptr addrspace(4)
  call void @llvm.lifetime.start.p0(i64 64, ptr nonnull %__SYCLKernel)
  %agg.tmp.sroa.0.sroa.0.0.copyload = load i64, ptr %_arg_accA1, align 8
  %agg.tmp.sroa.0.sroa.2.0._arg_accA1.ascast.sroa_idx = getelementptr inbounds i8, ptr %_arg_accA1, i64 8
  %agg.tmp.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp.sroa.0.sroa.2.0._arg_accA1.ascast.sroa_idx, align 8
  %agg.tmp5.sroa.0.sroa.0.0.copyload = load i64, ptr %_arg_accA2, align 8
  %agg.tmp5.sroa.0.sroa.2.0._arg_accA2.ascast.sroa_idx = getelementptr inbounds i8, ptr %_arg_accA2, i64 8
  %agg.tmp5.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp5.sroa.0.sroa.2.0._arg_accA2.ascast.sroa_idx, align 8
  %agg.tmp6.sroa.0.sroa.0.0.copyload = load i64, ptr %_arg_accA3, align 8
  %agg.tmp6.sroa.0.sroa.2.0._arg_accA3.ascast.sroa_idx = getelementptr inbounds i8, ptr %_arg_accA3, i64 8
  %agg.tmp6.sroa.0.sroa.2.0.copyload = load i64, ptr %agg.tmp6.sroa.0.sroa.2.0._arg_accA3.ascast.sroa_idx, align 8
  %0 = getelementptr inbounds %"class.sycl::_V1::accessor", ptr %__SYCLKernel, i64 0, i32 1
  store i64 %agg.tmp6.sroa.0.sroa.0.0.copyload, ptr %__SYCLKernel, align 8
  %AccessRange.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::detail::AccessorImplDevice", ptr %__SYCLKernel, i64 0, i32 1
  store i64 %agg.tmp.sroa.0.sroa.0.0.copyload, ptr %AccessRange.i.i.i.i.i, align 8
  %MemRange.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::detail::AccessorImplDevice", ptr %__SYCLKernel, i64 0, i32 2
  store i64 %agg.tmp5.sroa.0.sroa.0.0.copyload, ptr %MemRange.i.i.i.i.i, align 8
  %arrayidx.i21.i.i.i.i = getelementptr inbounds [2 x i64], ptr %__SYCLKernel, i64 0, i64 1
  store i64 %agg.tmp6.sroa.0.sroa.2.0.copyload, ptr %arrayidx.i21.i.i.i.i, align 8
  %arrayidx.i25.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::detail::AccessorImplDevice", ptr %__SYCLKernel, i64 0, i32 1, i32 0, i32 0, i64 1
  store i64 %agg.tmp.sroa.0.sroa.2.0.copyload, ptr %arrayidx.i25.i.i.i.i, align 8
  %arrayidx.i29.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::detail::AccessorImplDevice", ptr %__SYCLKernel, i64 0, i32 2, i32 0, i32 0, i64 1
  store i64 %agg.tmp5.sroa.0.sroa.2.0.copyload, ptr %arrayidx.i29.i.i.i.i, align 8
  %mul.i6.i.i.i.i = mul i64 %agg.tmp6.sroa.0.sroa.0.0.copyload, %agg.tmp5.sroa.0.sroa.2.0.copyload
  %1 = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %_arg_accA, i64 %mul.i6.i.i.i.i
  %add.ptr.i = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %1, i64 %agg.tmp6.sroa.0.sroa.2.0.copyload
  store ptr addrspace(1) %add.ptr.i, ptr %0, align 8
  %2 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, i64 8), align 8
  %3 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %4 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, i64 8), align 8
  %5 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32
  %ref.tmp6.ascast.i = addrspacecast ptr %ref.tmp6.i to ptr addrspace(4)
  %cmp.i11 = icmp ult i64 %2, 2147483648
  %cmp.i = icmp ult i64 %3, 2147483648
  %cmp.i15 = icmp ult i64 %4, 2147483648
  %sub.i = sub nsw i64 %2, %4
  %cmp.i12 = icmp ult i64 %5, 2147483648
  %sub5.i = sub nsw i64 %3, %5
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %ref.tmp6.i)
  store float 5.000000e+00, ptr %ref.tmp6.i, align 4
  %call.i.i = call spir_func noundef zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4) %ref.tmp6.ascast.i)
  call void @llvm.lifetime.start.p0(i64 2, ptr nonnull %agg.tmp.i17)
  store i16 %call.i.i, ptr %agg.tmp.i17, align 2
  %call.i18 = call spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) @_Z26__spirv_CompositeConstruct(ptr noundef nonnull byval(%"class.sycl::_V1::ext::oneapi::bfloat16") align 2 %agg.tmp.i17)
  call void @llvm.lifetime.end.p0(i64 2, ptr nonnull %agg.tmp.i17)
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %ref.tmp6.i)
  %lambda.i = getelementptr inbounds %class.anon.0, ptr addrspace(4) %__SYCLKernel.ascast, i64 0, i32 1
  %ref.tmp.ascast.i21 = addrspacecast ptr %ref.tmp.i20 to ptr addrspace(4)
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %ref.tmp.i20)
  store ptr addrspace(4) %lambda.i, ptr %ref.tmp.i20, align 8
  %call.i22 = call spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) @_Z43__spirv_CooperativeMatrixApplyFunctionINTEL(ptr addrspace(4) noundef align 8 dereferenceable(8) %ref.tmp.ascast.i21, target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) noundef %call.i18)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %ref.tmp.i20)
  %6 = load ptr addrspace(1), ptr %0, align 8
  %7 = load i64, ptr %__SYCLKernel, align 8
  %8 = load i64, ptr %arrayidx.i29.i.i.i.i, align 8
  %mul.i6.i.i.i.i.i = mul i64 %7, %8
  %9 = load i64, ptr %arrayidx.i21.i.i.i.i, align 8
  %add.i7.i.i.i.i.i = add i64 %mul.i6.i.i.i.i.i, %9
  %idx.neg.i.i = sub i64 0, %add.i7.i.i.i.i.i
  %add.ptr.i.i = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %6, i64 %idx.neg.i.i
  %mul12.i = shl nsw i64 %sub.i, 8
  %add.ptr.i43 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %add.ptr.i.i, i64 %mul12.i
  %div14.i = and i64 %sub5.i, -16
  %add.ptr.i44 = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %add.ptr.i43, i64 %div14.i
  call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHRPU3AS4iPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_3ili(ptr addrspace(1) noundef %add.ptr.i44, target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) noundef %call.i22, i32 noundef 0, i64 noundef 0)
  call void @llvm.lifetime.end.p0(i64 64, ptr nonnull %__SYCLKernel)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) @_Z26__spirv_CompositeConstruct(ptr noundef byval(%"class.sycl::_V1::ext::oneapi::bfloat16") align 2) local_unnamed_addr

; Function Attrs: convergent nounwind
declare dso_local spir_func zeroext i16 @__devicelib_ConvertFToBF16INTEL(ptr addrspace(4) noundef align 4 dereferenceable(4)) local_unnamed_addr

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) @_Z43__spirv_CooperativeMatrixApplyFunctionINTEL(ptr addrspace(4) noundef align 8 dereferenceable(8), target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) noundef) local_unnamed_addr

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z33__spirv_CooperativeMatrixStoreKHRPU3AS4iPU3AS144__spirv_CooperativeMatrixKHR__uint_3_12_12_3ili(ptr addrspace(1) noundef, target("spirv.CooperativeMatrixKHR", i16, 3, 8, 16, 0) noundef, i32 noundef, i64 noundef) local_unnamed_addr

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 18.0.0 (https://github.com/intel/llvm.git)"}
