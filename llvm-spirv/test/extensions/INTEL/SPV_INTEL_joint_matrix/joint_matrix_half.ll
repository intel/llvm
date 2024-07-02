; compiled from joint_matrix_half.cpp test from intel/llvm

; RUN: llvm-as < %s -o %t.bc

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_bfloat16_conversion,+SPV_INTEL_joint_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: TypeInt [[#INT:]] 32
; CHECK-SPIRV-DAG: TypeFloat [[#Half:]] 16
; CHECK-SPIRV-DAG: TypeFloat [[#Float:]] 32
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST8:]] 8
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST16:]] 16
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST3:]] 3
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST2:]] 2
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST1:]] 1
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST0:]] 0
; CHECK-SPIRV-DAG: TypeJointMatrixINTEL [[#MatTy1:]] [[#Float]] [[#CONST8]] [[#CONST16]] [[#CONST3]] [[#CONST3]] [[#CONST2]]
; CHECK-SPIRV-DAG: TypeJointMatrixINTEL [[#MatTy2:]] [[#Half]] [[#CONST8]] [[#CONST16]] [[#CONST0]] [[#CONST3]] [[#CONST0]]
; CHECK-SPIRV-DAG: TypeJointMatrixINTEL [[#MatTy3:]] [[#Half]] [[#CONST16]] [[#CONST16]] [[#CONST2]] [[#CONST3]] [[#CONST1]]

; CHECK-LLVM: call spir_func target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z80__spirv_JointMatrixLoadINTEL_RPU3AS142__spirv_JointMatrixINTEL__float_8_16_3_3_2PU3AS1fliii(ptr addrspace(1) %{{.*}}, i64 %{{.*}}, i32 0, i32 3, i32 0)
; CHECK-LLVM: call spir_func target("spirv.JointMatrixINTEL", half, 8, 16, 0, 3, 0) @"_Z79__spirv_JointMatrixLoadINTEL_RPU3AS141__spirv_JointMatrixINTEL__half_8_16_0_3_0PU3AS140class.sycl::_V1::detail::half_impl::halfliii"(ptr addrspace(1) %{{.*}}, i64 %{{.*}}, i32 0, i32 3, i32 0)
; CHECK-LLVM: call spir_func target("spirv.JointMatrixINTEL", half, 16, 16, 2, 3, 1) @"_Z80__spirv_JointMatrixLoadINTEL_RPU3AS142__spirv_JointMatrixINTEL__half_16_16_2_3_1PU3AS140class.sycl::_V1::detail::half_impl::halfliii"(ptr addrspace(1) %{{.*}}, i64 %{{.*}}, i32 2, i32 3, i32 0)
; CHECK-LLVM: call spir_func target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z27__spirv_JointMatrixMadINTELPU3AS141__spirv_JointMatrixINTEL__half_8_16_0_3_0PU3AS142__spirv_JointMatrixINTEL__half_16_16_2_3_1PU3AS142__spirv_JointMatrixINTEL__float_8_16_3_3_2i(target("spirv.JointMatrixINTEL", half, 8, 16, 0, 3, 0) %{{.*}}, target("spirv.JointMatrixINTEL", half, 16, 16, 2, 3, 1) %{{.*}}, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) %{{.*}}, i32 3)
; CHECK-LLVM: call spir_func void @_Z29__spirv_JointMatrixStoreINTELPU3AS1fPU3AS142__spirv_JointMatrixINTEL__float_8_16_3_3_2liii(ptr addrspace(1) %{{.*}}, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) %{{.*}}, i64 %{{.*}}, i32 0, i32 3, i32 0)

; ModuleID = 'half.bc'
source_filename = "../joint_matrix_half.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::detail::half_impl::half" = type { half }

$_ZTSZZ15matrix_multiplyIfN4sycl3_V16detail9half_impl4halfELm16ELm32ELm16ELm64ELm16ELm32EEvR10big_matrixIT_XT5_EXT6_EERS5_IT0_XT1_EXT2_EERS5_IS9_XT3_EXT4_EEENKUlRNS1_7handlerEE_clESF_E7imatrix = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiplyIfN4sycl3_V16detail9half_impl4halfELm16ELm32ELm16ELm64ELm16ELm32EEvR10big_matrixIT_XT5_EXT6_EERS5_IT0_XT1_EXT2_EERS5_IS9_XT3_EXT4_EEENKUlRNS1_7handlerEE_clESF_E7imatrix(ptr addrspace(1) noundef align 2 %_arg_accA, ptr addrspace(1) noundef align 2 %_arg_accB, ptr addrspace(1) noundef align 4 %_arg_accC, i64 noundef %_arg_N, i64 noundef %_arg_K) local_unnamed_addr #0 comdat !srcloc !49 !kernel_arg_buffer_location !50 !kernel_arg_runtime_aligned !51 !kernel_arg_exclusive_ptr !51 !intel_reqd_sub_group_size !52 !sycl_used_aspects !53 !sycl_fixed_targets !54 !sycl_kernel_omit_args !55 {
entry:
  call void @__itt_offload_wi_start_wrapper()
  %0 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, i64 8), align 8, !noalias !56
  %1 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32, !noalias !56
  %2 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, i64 8), align 8, !noalias !63
  %3 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32, !noalias !63
  %cmp.i.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %cmp.i61.i = icmp ult i64 %1, 2147483648
  tail call void @llvm.assume(i1 %cmp.i61.i)
  %cmp.i63.i = icmp ult i64 %2, 2147483648
  tail call void @llvm.assume(i1 %cmp.i63.i)
  %sub.i = sub nsw i64 %0, %2
  %cmp.i66.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i66.i)
  %sub5.i = sub nsw i64 %1, %3
  %mul.i = shl nsw i64 %sub.i, 3
  %mul8.i = mul i64 %mul.i, %_arg_N
  %add.ptr.i.i = getelementptr inbounds float, ptr addrspace(1) %_arg_accC, i64 %mul8.i
  %div58.i = and i64 %sub5.i, -16
  %add.ptr.i80.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i.i, i64 %div58.i
  %call1.i.i = tail call spir_func noundef target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z28__spirv_JointMatrixLoadINTELIU3AS1ffLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEPNS1_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS3_S5_i(ptr addrspace(1) noundef %add.ptr.i80.i, i64 noundef %_arg_N, i32 noundef 0, i32 noundef 3, i32 noundef 0) #3
  %mul34.i = shl nsw i64 %div58.i, 1
  %div1159.i = lshr i64 %_arg_K, 4
  %mul18.i = mul i64 %mul.i, %_arg_K
  %add.ptr.i95.i = getelementptr inbounds %"class.sycl::_V1::detail::half_impl::half", ptr addrspace(1) %_arg_accA, i64 %mul18.i
  %mul30.i = shl i64 %_arg_N, 1
  %invariant.gep = getelementptr %"class.sycl::_V1::detail::half_impl::half", ptr addrspace(1) %_arg_accB, i64 %mul34.i
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.body.i, %entry
  %sub_c.sroa.0.0.i = phi target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) [ %call1.i.i, %entry ], [ %call.i.i, %for.body.i ]
  %k.0.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  %conv.i = zext i32 %k.0.i to i64
  %cmp.i = icmp ugt i64 %div1159.i, %conv.i
  br i1 %cmp.i, label %for.body.i, label %_ZZZ15matrix_multiplyIfN4sycl3_V16detail9half_impl4halfELm16ELm32ELm16ELm64ELm16ELm32EEvR10big_matrixIT_XT5_EXT6_EERS5_IT0_XT1_EXT2_EERS5_IS9_XT3_EXT4_EEENKUlRNS1_7handlerEE_clESF_ENKUlNS1_7nd_itemILi2EEEE_clESI_.exit

for.body.i:                                       ; preds = %for.cond.i
  %mul19.i = shl nsw i32 %k.0.i, 4
  %conv20.i = zext i32 %mul19.i to i64
  %add.ptr.i96.i = getelementptr inbounds %"class.sycl::_V1::detail::half_impl::half", ptr addrspace(1) %add.ptr.i95.i, i64 %conv20.i
  %call1.i74.i = tail call spir_func noundef target("spirv.JointMatrixINTEL", half, 8, 16, 0, 3, 0) @_Z28__spirv_JointMatrixLoadINTELIU3AS1N4sycl3_V16detail9half_impl4halfES4_Lm8ELm16ELN5__spv9MatrixUseE0ELNS6_12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef %add.ptr.i96.i, i64 noundef %_arg_K, i32 noundef 0, i32 noundef 3, i32 noundef 0) #3
  %div27.i = shl nsw i32 %k.0.i, 3
  %conv28.i = zext i32 %div27.i to i64
  %mul31.i = mul i64 %mul30.i, %conv28.i
  %gep = getelementptr %"class.sycl::_V1::detail::half_impl::half", ptr addrspace(1) %invariant.gep, i64 %mul31.i
  %call1.i78.i = tail call spir_func noundef target("spirv.JointMatrixINTEL", half, 16, 16, 2, 3, 1) @_Z28__spirv_JointMatrixLoadINTELIU3AS1N4sycl3_V16detail9half_impl4halfES4_Lm16ELm16ELN5__spv9MatrixUseE1ELNS6_12MatrixLayoutE2ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef %gep, i64 noundef %mul30.i, i32 noundef 2, i32 noundef 3, i32 noundef 0) #3
  %call.i.i = tail call spir_func noundef target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z27__spirv_JointMatrixMadINTELIN4sycl3_V16detail9half_impl4halfEfLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS6_1ELS6_2ELNS5_12MatrixLayoutE0ELS7_2ELS7_3ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT9_EXT10_EXT6_EEEPNSA_IT_XT1_EXT2_EXT7_EXT10_EXT4_EEEPNSA_ISE_XT2_EXT3_EXT8_EXT10_EXT5_EEESD_S9_(target("spirv.JointMatrixINTEL", half, 8, 16, 0, 3, 0) noundef %call1.i74.i, target("spirv.JointMatrixINTEL", half, 16, 16, 2, 3, 1) noundef %call1.i78.i, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) noundef %sub_c.sroa.0.0.i, i32 noundef 3) #3, !noalias !68
  %add.i = add nuw nsw i32 %k.0.i, 1
  br label %for.cond.i, !llvm.loop !71

_ZZZ15matrix_multiplyIfN4sycl3_V16detail9half_impl4halfELm16ELm32ELm16ELm64ELm16ELm32EEvR10big_matrixIT_XT5_EXT6_EERS5_IT0_XT1_EXT2_EERS5_IS9_XT3_EXT4_EEENKUlRNS1_7handlerEE_clESF_ENKUlNS1_7nd_itemILi2EEEE_clESI_.exit: ; preds = %for.cond.i
  tail call spir_func void @_Z29__spirv_JointMatrixStoreINTELIU3AS1ffLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEvPT_PNS1_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEmS3_S5_i(ptr addrspace(1) noundef %add.ptr.i80.i, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) noundef %sub_c.sroa.0.0.i, i64 noundef %_arg_N, i32 noundef 0, i32 noundef 3, i32 noundef 0) #3
  call void @__itt_offload_wi_finish_wrapper()
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z28__spirv_JointMatrixLoadINTELIU3AS1ffLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEPNS1_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS3_S5_i(ptr addrspace(1) noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.JointMatrixINTEL", half, 8, 16, 0, 3, 0) @_Z28__spirv_JointMatrixLoadINTELIU3AS1N4sycl3_V16detail9half_impl4halfES4_Lm8ELm16ELN5__spv9MatrixUseE0ELNS6_12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.JointMatrixINTEL", half, 16, 16, 2, 3, 1) @_Z28__spirv_JointMatrixLoadINTELIU3AS1N4sycl3_V16detail9half_impl4halfES4_Lm16ELm16ELN5__spv9MatrixUseE1ELNS6_12MatrixLayoutE2ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z27__spirv_JointMatrixMadINTELIN4sycl3_V16detail9half_impl4halfEfLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS6_1ELS6_2ELNS5_12MatrixLayoutE0ELS7_2ELS7_3ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT9_EXT10_EXT6_EEEPNSA_IT_XT1_EXT2_EXT7_EXT10_EXT4_EEEPNSA_ISE_XT2_EXT3_EXT8_EXT10_EXT5_EEESD_S9_(target("spirv.JointMatrixINTEL", half, 8, 16, 0, 3, 0) noundef, target("spirv.JointMatrixINTEL", half, 16, 16, 2, 3, 1) noundef, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z29__spirv_JointMatrixStoreINTELIU3AS1ffLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEvPT_PNS1_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEmS3_S5_i(ptr addrspace(1) noundef, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)

declare void @__itt_offload_wi_start_wrapper()

declare void @__itt_offload_wi_finish_wrapper()

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../joint_matrix_half.cpp" "sycl-optlevel"="2" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!sycl_types_that_use_aspects = !{!4}
!sycl_aspects = !{!5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47}
!llvm.ident = !{!48}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"class.sycl::_V1::detail::half_impl::half", i32 5}
!5 = !{!"cpu", i32 1}
!6 = !{!"gpu", i32 2}
!7 = !{!"accelerator", i32 3}
!8 = !{!"custom", i32 4}
!9 = !{!"fp16", i32 5}
!10 = !{!"fp64", i32 6}
!11 = !{!"image", i32 9}
!12 = !{!"online_compiler", i32 10}
!13 = !{!"online_linker", i32 11}
!14 = !{!"queue_profiling", i32 12}
!15 = !{!"usm_device_allocations", i32 13}
!16 = !{!"usm_host_allocations", i32 14}
!17 = !{!"usm_shared_allocations", i32 15}
!18 = !{!"usm_system_allocations", i32 17}
!19 = !{!"ext_intel_pci_address", i32 18}
!20 = !{!"ext_intel_gpu_eu_count", i32 19}
!21 = !{!"ext_intel_gpu_eu_simd_width", i32 20}
!22 = !{!"ext_intel_gpu_slices", i32 21}
!23 = !{!"ext_intel_gpu_subslices_per_slice", i32 22}
!24 = !{!"ext_intel_gpu_eu_count_per_subslice", i32 23}
!25 = !{!"ext_intel_max_mem_bandwidth", i32 24}
!26 = !{!"ext_intel_mem_channel", i32 25}
!27 = !{!"usm_atomic_host_allocations", i32 26}
!28 = !{!"usm_atomic_shared_allocations", i32 27}
!29 = !{!"atomic64", i32 28}
!30 = !{!"ext_intel_device_info_uuid", i32 29}
!31 = !{!"ext_oneapi_srgb", i32 30}
!32 = !{!"ext_oneapi_native_assert", i32 31}
!33 = !{!"host_debuggable", i32 32}
!34 = !{!"ext_intel_gpu_hw_threads_per_eu", i32 33}
!35 = !{!"ext_oneapi_cuda_async_barrier", i32 34}
!36 = !{!"ext_oneapi_bfloat16_math_functions", i32 35}
!37 = !{!"ext_intel_free_memory", i32 36}
!38 = !{!"ext_intel_device_id", i32 37}
!39 = !{!"ext_intel_memory_clock_rate", i32 38}
!40 = !{!"ext_intel_memory_bus_width", i32 39}
!41 = !{!"emulated", i32 40}
!42 = !{!"ext_intel_legacy_image", i32 41}
!43 = !{!"int64_base_atomics", i32 7}
!44 = !{!"int64_extended_atomics", i32 8}
!45 = !{!"usm_system_allocator", i32 17}
!46 = !{!"usm_restricted_shared_allocations", i32 16}
!47 = !{!"host", i32 0}
!48 = !{!"clang version 17.0.0 (https://github.com/intel/llvm.git 93f477358d74ae90024f758e7eeb97d4b13cea42)"}
!49 = !{i32 10643216}
!50 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!51 = !{i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false}
!52 = !{i32 16}
!53 = !{i32 5}
!54 = !{}
!55 = !{i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false}
!56 = !{!57, !59, !61}
!57 = distinct !{!57, !58, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv: %agg.result"}
!58 = distinct !{!58, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv"}
!59 = distinct !{!59, !60, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN4sycl3_V12idILi2EEEEET0_v: %agg.result"}
!60 = distinct !{!60, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN4sycl3_V12idILi2EEEEET0_v"}
!61 = distinct !{!61, !62, !"_ZN4sycl3_V16detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_: %agg.result"}
!62 = distinct !{!62, !"_ZN4sycl3_V16detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_"}
!63 = !{!64, !66, !61}
!64 = distinct !{!64, !65, !"_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv: %agg.result"}
!65 = distinct !{!65, !"_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv"}
!66 = distinct !{!66, !67, !"_ZN7__spirvL21initLocalInvocationIdILi2EN4sycl3_V12idILi2EEEEET0_v: %agg.result"}
!67 = distinct !{!67, !"_ZN7__spirvL21initLocalInvocationIdILi2EN4sycl3_V12idILi2EEEEET0_v"}
!68 = !{!69}
!69 = distinct !{!69, !70, !"_ZN4sycl3_V13ext6oneapi12experimental6matrix16joint_matrix_madINS0_9sub_groupENS0_6detail9half_impl4halfES9_fLm8ELm16ELm16ELNS4_6layoutE0ELSA_2EEENS4_12joint_matrixIT_T2_LNS4_3useE2EXT3_EXT5_ELSA_3EEESC_RNSB_ISC_T0_LSE_0EXT3_EXT4_EXT6_EEERNSB_ISC_T1_LSE_1EXT4_EXT5_EXT7_EEERSF_: %agg.result"}
!70 = distinct !{!70, !"_ZN4sycl3_V13ext6oneapi12experimental6matrix16joint_matrix_madINS0_9sub_groupENS0_6detail9half_impl4halfES9_fLm8ELm16ELm16ELNS4_6layoutE0ELSA_2EEENS4_12joint_matrixIT_T2_LNS4_3useE2EXT3_EXT5_ELSA_3EEESC_RNSB_ISC_T0_LSE_0EXT3_EXT4_EXT6_EEERNSB_ISC_T1_LSE_1EXT4_EXT5_EXT7_EEERSF_"}
!71 = distinct !{!71, !72}
!72 = !{!"llvm.loop.mustprogress"}
