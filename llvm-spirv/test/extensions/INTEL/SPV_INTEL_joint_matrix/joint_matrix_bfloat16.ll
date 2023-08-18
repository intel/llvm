; compiled from joint_matrix_bfloat16.cpp test from intel/llvm

; RUN: llvm-as < %s -o %t.bc

; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_bfloat16_conversion,+SPV_INTEL_joint_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: TypeInt [[#SHORT:]] 16
; CHECK-SPIRV-DAG: TypeInt [[#INT:]] 32
; CHECK-SPIRV-DAG: TypeFloat [[#Float:]] 32
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST8:]] 8
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST16:]] 16
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST3:]] 3
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST2:]] 2
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST1:]] 1
; CHECK-SPIRV-DAG: Constant [[#INT]] [[#CONST0:]] 0
; CHECK-SPIRV-DAG: TypeJointMatrixINTEL [[#MatTy1:]] [[#Float]] [[#CONST8]] [[#CONST16]] [[#CONST3]] [[#CONST3]] [[#CONST2]]
; CHECK-SPIRV-DAG: TypeJointMatrixINTEL [[#MatTy2:]] [[#SHORT]] [[#CONST8]] [[#CONST16]] [[#CONST0]] [[#CONST3]] [[#CONST0]] [[#CONST1]]
; CHECK-SPIRV-DAG: TypeJointMatrixINTEL [[#MatTy3:]] [[#SHORT]] [[#CONST16]] [[#CONST16]] [[#CONST2]] [[#CONST3]] [[#CONST1]] [[#CONST1]]

; CHECK-LLVM: call spir_func target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z80__spirv_JointMatrixLoadINTEL_RPU3AS142__spirv_JointMatrixINTEL__float_8_16_3_3_2PU3AS1fliii(ptr addrspace(1) %{{.*}}, i64 32, i32 0, i32 3, i32 0)
; CHECK-LLVM: call spir_func target("spirv.JointMatrixINTEL", i16, 8, 16, 0, 3, 0, 1) @"_Z82__spirv_JointMatrixLoadINTEL_RPU3AS144__spirv_JointMatrixINTEL__short_8_16_0_3_0_1PU3AS138class.sycl::_V1::ext::oneapi::bfloat16liii"(ptr addrspace(1) %{{.*}}, i64 32, i32 0, i32 3, i32 0)
; CHECK-LLVM: call spir_func target("spirv.JointMatrixINTEL", i16, 16, 16, 2, 3, 1, 1) @"_Z83__spirv_JointMatrixLoadINTEL_RPU3AS145__spirv_JointMatrixINTEL__short_16_16_2_3_1_1PU3AS138class.sycl::_V1::ext::oneapi::bfloat16liii"(ptr addrspace(1) %{{.*}}, i64 64, i32 2, i32 3, i32 0)
; CHECK-LLVM: call spir_func target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z27__spirv_JointMatrixMadINTELPU3AS144__spirv_JointMatrixINTEL__short_8_16_0_3_0_1PU3AS145__spirv_JointMatrixINTEL__short_16_16_2_3_1_1PU3AS142__spirv_JointMatrixINTEL__float_8_16_3_3_2i(target("spirv.JointMatrixINTEL", i16, 8, 16, 0, 3, 0, 1) %{{.*}}, target("spirv.JointMatrixINTEL", i16, 16, 16, 2, 3, 1, 1) %{{.*}}, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) %{{.*}}, i32 3)
; CHECK-LLVM: call spir_func void @_Z29__spirv_JointMatrixStoreINTELPU3AS1fPU3AS142__spirv_JointMatrixINTEL__float_8_16_3_3_2liii(ptr addrspace(1) %{{.*}}, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) %{{.*}}, i64 32, i32 0, i32 3, i32 0)

; ModuleID = 'joint_matrix_bfloat16-sycl-spir64-unknown-unknown.bc'
source_filename = "../joint_matrix_bfloat16.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }

$_ZTSZZ15matrix_multiplyIfN4sycl3_V13ext6oneapi8bfloat16ELm16ELm32ELm32EEvR10big_matrixIT_XT1_EXT2_EERS5_IT0_XT1_EXT3_EERS5_IS9_XdvT3_Li2EEXmlT2_Li2EEEENKUlRNS1_7handlerEE_clESF_E7imatrix = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ15matrix_multiplyIfN4sycl3_V13ext6oneapi8bfloat16ELm16ELm32ELm32EEvR10big_matrixIT_XT1_EXT2_EERS5_IT0_XT1_EXT3_EERS5_IS9_XdvT3_Li2EEXmlT2_Li2EEEENKUlRNS1_7handlerEE_clESF_E7imatrix(ptr addrspace(1) noundef align 4 %_arg_accC, ptr addrspace(1) noundef align 2 %_arg_accA, ptr addrspace(1) noundef align 2 %_arg_accB) local_unnamed_addr #0 comdat !srcloc !48 !kernel_arg_buffer_location !49 !kernel_arg_runtime_aligned !50 !kernel_arg_exclusive_ptr !50 !intel_reqd_sub_group_size !51 !sycl_fixed_targets !52 !sycl_kernel_omit_args !53 {
entry:
  call void @__itt_offload_wi_start_wrapper()
  %0 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, i64 8), align 8, !noalias !54
  %1 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32, !noalias !54
  %2 = load i64, ptr addrspace(1) getelementptr inbounds (i8, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, i64 8), align 8, !noalias !61
  %3 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32, !noalias !61
  %cmp.i.i = icmp ult i64 %0, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  %cmp.i50.i = icmp ult i64 %1, 2147483648
  tail call void @llvm.assume(i1 %cmp.i50.i)
  %cmp.i52.i = icmp ult i64 %2, 2147483648
  tail call void @llvm.assume(i1 %cmp.i52.i)
  %sub.i = sub nsw i64 %0, %2
  %cmp.i55.i = icmp ult i64 %3, 2147483648
  tail call void @llvm.assume(i1 %cmp.i55.i)
  %sub5.i = sub nsw i64 %1, %3
  %mul8.i = shl nsw i64 %sub.i, 8
  %add.ptr.i.i = getelementptr inbounds float, ptr addrspace(1) %_arg_accC, i64 %mul8.i
  %div48.i = and i64 %sub5.i, -16
  %add.ptr.i69.i = getelementptr inbounds float, ptr addrspace(1) %add.ptr.i.i, i64 %div48.i
  %call1.i.i = tail call spir_func noundef target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z28__spirv_JointMatrixLoadINTELIU3AS1ffLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEPNS1_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS3_S5_i(ptr addrspace(1) noundef %add.ptr.i69.i, i64 noundef 32, i32 noundef 0, i32 noundef 3, i32 noundef 0) #3
  %mul28.i = shl nsw i64 %div48.i, 1
  %add.ptr.i84.i = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %_arg_accA, i64 %mul8.i
  %invariant.gep = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %_arg_accB, i64 %mul28.i
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.body.i, %entry
  %sub_c.sroa.0.0.i = phi target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) [ %call1.i.i, %entry ], [ %call.i.i, %for.body.i ]
  %k.0.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  %cmp.i = icmp ult i32 %k.0.i, 2
  br i1 %cmp.i, label %for.body.i, label %_ZZZ15matrix_multiplyIfN4sycl3_V13ext6oneapi8bfloat16ELm16ELm32ELm32EEvR10big_matrixIT_XT1_EXT2_EERS5_IT0_XT1_EXT3_EERS5_IS9_XdvT3_Li2EEXmlT2_Li2EEEENKUlRNS1_7handlerEE_clESF_ENKUlNS1_7nd_itemILi2EEEE_clESI_.exit

for.body.i:                                       ; preds = %for.cond.i
  %mul16.i = shl nuw nsw i32 %k.0.i, 4
  %conv17.i = zext i32 %mul16.i to i64
  %add.ptr.i85.i = getelementptr inbounds %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %add.ptr.i84.i, i64 %conv17.i
  %call1.i63.i = tail call spir_func noundef target("spirv.JointMatrixINTEL", i16, 8, 16, 0, 3, 0, 1) @_Z28__spirv_JointMatrixLoadINTELIU3AS1N4sycl3_V13ext6oneapi8bfloat16ES4_Lm8ELm16ELN5__spv9MatrixUseE0ELNS6_12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef %add.ptr.i85.i, i64 noundef 32, i32 noundef 0, i32 noundef 3, i32 noundef 0) #3
  %div23.i = shl nuw nsw i32 %k.0.i, 3
  %conv24.i = zext i32 %div23.i to i64
  %mul25.i = shl nuw nsw i64 %conv24.i, 6
  %gep = getelementptr %"class.sycl::_V1::ext::oneapi::bfloat16", ptr addrspace(1) %invariant.gep, i64 %mul25.i
  %call1.i67.i = tail call spir_func noundef target("spirv.JointMatrixINTEL", i16, 16, 16, 2, 3, 1, 1) @_Z28__spirv_JointMatrixLoadINTELIU3AS1N4sycl3_V13ext6oneapi8bfloat16ES4_Lm16ELm16ELN5__spv9MatrixUseE1ELNS6_12MatrixLayoutE2ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef %gep, i64 noundef 64, i32 noundef 2, i32 noundef 3, i32 noundef 0) #3
  %call.i.i = tail call spir_func noundef target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z27__spirv_JointMatrixMadINTELIN4sycl3_V13ext6oneapi8bfloat16EfLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS6_1ELS6_2ELNS5_12MatrixLayoutE0ELS7_2ELS7_3ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT9_EXT10_EXT6_EEEPNSA_IT_XT1_EXT2_EXT7_EXT10_EXT4_EEEPNSA_ISE_XT2_EXT3_EXT8_EXT10_EXT5_EEESD_S9_(target("spirv.JointMatrixINTEL", i16, 8, 16, 0, 3, 0, 1) noundef %call1.i63.i, target("spirv.JointMatrixINTEL", i16, 16, 16, 2, 3, 1, 1) noundef %call1.i67.i, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) noundef %sub_c.sroa.0.0.i, i32 noundef 3) #3, !noalias !66
  %add.i = add nuw nsw i32 %k.0.i, 1
  br label %for.cond.i, !llvm.loop !69

_ZZZ15matrix_multiplyIfN4sycl3_V13ext6oneapi8bfloat16ELm16ELm32ELm32EEvR10big_matrixIT_XT1_EXT2_EERS5_IT0_XT1_EXT3_EERS5_IS9_XdvT3_Li2EEXmlT2_Li2EEEENKUlRNS1_7handlerEE_clESF_ENKUlNS1_7nd_itemILi2EEEE_clESI_.exit: ; preds = %for.cond.i
  tail call spir_func void @_Z29__spirv_JointMatrixStoreINTELIU3AS1ffLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEvPT_PNS1_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEmS3_S5_i(ptr addrspace(1) noundef %add.ptr.i69.i, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) noundef %sub_c.sroa.0.0.i, i64 noundef 32, i32 noundef 0, i32 noundef 3, i32 noundef 0) #3
  call void @__itt_offload_wi_finish_wrapper()
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z28__spirv_JointMatrixLoadINTELIU3AS1ffLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEPNS1_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS3_S5_i(ptr addrspace(1) noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.JointMatrixINTEL", i16, 8, 16, 0, 3, 0, 1) @_Z28__spirv_JointMatrixLoadINTELIU3AS1N4sycl3_V13ext6oneapi8bfloat16ES4_Lm8ELm16ELN5__spv9MatrixUseE0ELNS6_12MatrixLayoutE0ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.JointMatrixINTEL", i16, 16, 16, 2, 3, 1, 1) @_Z28__spirv_JointMatrixLoadINTELIU3AS1N4sycl3_V13ext6oneapi8bfloat16ES4_Lm16ELm16ELN5__spv9MatrixUseE1ELNS6_12MatrixLayoutE2ELNS6_5Scope4FlagE3EEPNS6_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEPT_mS8_SA_i(ptr addrspace(1) noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local spir_func noundef target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) @_Z27__spirv_JointMatrixMadINTELIN4sycl3_V13ext6oneapi8bfloat16EfLm8ELm16ELm16ELN5__spv9MatrixUseE0ELS6_1ELS6_2ELNS5_12MatrixLayoutE0ELS7_2ELS7_3ELNS5_5Scope4FlagE3EEPNS5_24__spirv_JointMatrixINTELIT0_XT1_EXT3_EXT9_EXT10_EXT6_EEEPNSA_IT_XT1_EXT2_EXT7_EXT10_EXT4_EEEPNSA_ISE_XT2_EXT3_EXT8_EXT10_EXT5_EEESD_S9_(target("spirv.JointMatrixINTEL", i16, 8, 16, 0, 3, 0, 1) noundef, target("spirv.JointMatrixINTEL", i16, 16, 16, 2, 3, 1, 1) noundef, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z29__spirv_JointMatrixStoreINTELIU3AS1ffLm8ELm16ELN5__spv9MatrixUseE2ELNS1_12MatrixLayoutE3ELNS1_5Scope4FlagE3EEvPT_PNS1_24__spirv_JointMatrixINTELIT0_XT1_EXT2_EXT4_EXT5_EXT3_EEEmS3_S5_i(ptr addrspace(1) noundef, target("spirv.JointMatrixINTEL", float, 8, 16, 3, 3, 2) noundef, i64 noundef, i32 noundef, i32 noundef, i32 noundef) local_unnamed_addr #2

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)

declare void @__itt_offload_wi_start_wrapper()

declare void @__itt_offload_wi_finish_wrapper()

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../joint_matrix_bfloat16.cpp" "sycl-optlevel"="2" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #2 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent nounwind }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!sycl_aspects = !{!4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46}
!llvm.ident = !{!47}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"cpu", i32 1}
!5 = !{!"gpu", i32 2}
!6 = !{!"accelerator", i32 3}
!7 = !{!"custom", i32 4}
!8 = !{!"fp16", i32 5}
!9 = !{!"fp64", i32 6}
!10 = !{!"image", i32 9}
!11 = !{!"online_compiler", i32 10}
!12 = !{!"online_linker", i32 11}
!13 = !{!"queue_profiling", i32 12}
!14 = !{!"usm_device_allocations", i32 13}
!15 = !{!"usm_host_allocations", i32 14}
!16 = !{!"usm_shared_allocations", i32 15}
!17 = !{!"usm_system_allocations", i32 17}
!18 = !{!"ext_intel_pci_address", i32 18}
!19 = !{!"ext_intel_gpu_eu_count", i32 19}
!20 = !{!"ext_intel_gpu_eu_simd_width", i32 20}
!21 = !{!"ext_intel_gpu_slices", i32 21}
!22 = !{!"ext_intel_gpu_subslices_per_slice", i32 22}
!23 = !{!"ext_intel_gpu_eu_count_per_subslice", i32 23}
!24 = !{!"ext_intel_max_mem_bandwidth", i32 24}
!25 = !{!"ext_intel_mem_channel", i32 25}
!26 = !{!"usm_atomic_host_allocations", i32 26}
!27 = !{!"usm_atomic_shared_allocations", i32 27}
!28 = !{!"atomic64", i32 28}
!29 = !{!"ext_intel_device_info_uuid", i32 29}
!30 = !{!"ext_oneapi_srgb", i32 30}
!31 = !{!"ext_oneapi_native_assert", i32 31}
!32 = !{!"host_debuggable", i32 32}
!33 = !{!"ext_intel_gpu_hw_threads_per_eu", i32 33}
!34 = !{!"ext_oneapi_cuda_async_barrier", i32 34}
!35 = !{!"ext_oneapi_bfloat16_math_functions", i32 35}
!36 = !{!"ext_intel_free_memory", i32 36}
!37 = !{!"ext_intel_device_id", i32 37}
!38 = !{!"ext_intel_memory_clock_rate", i32 38}
!39 = !{!"ext_intel_memory_bus_width", i32 39}
!40 = !{!"emulated", i32 40}
!41 = !{!"ext_intel_legacy_image", i32 41}
!42 = !{!"int64_base_atomics", i32 7}
!43 = !{!"int64_extended_atomics", i32 8}
!44 = !{!"usm_system_allocator", i32 17}
!45 = !{!"usm_restricted_shared_allocations", i32 16}
!46 = !{!"host", i32 0}
!47 = !{!"clang version 17.0.0 (https://github.com/intel/llvm.git 93f477358d74ae90024f758e7eeb97d4b13cea42)"}
!48 = !{i32 10642943}
!49 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!50 = !{i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false}
!51 = !{i32 16}
!52 = !{}
!53 = !{i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true, i1 false, i1 true, i1 true, i1 true}
!54 = !{!55, !57, !59}
!55 = distinct !{!55, !56, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv: %agg.result"}
!56 = distinct !{!56, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv"}
!57 = distinct !{!57, !58, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN4sycl3_V12idILi2EEEEET0_v: %agg.result"}
!58 = distinct !{!58, !"_ZN7__spirvL22initGlobalInvocationIdILi2EN4sycl3_V12idILi2EEEEET0_v"}
!59 = distinct !{!59, !60, !"_ZN4sycl3_V16detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_: %agg.result"}
!60 = distinct !{!60, !"_ZN4sycl3_V16detail7Builder10getElementILi2EEEKNS0_7nd_itemIXT_EEEPS5_"}
!61 = !{!62, !64, !59}
!62 = distinct !{!62, !63, !"_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv: %agg.result"}
!63 = distinct !{!63, !"_ZN7__spirv28InitSizesSTLocalInvocationIdILi2EN4sycl3_V12idILi2EEEE8initSizeEv"}
!64 = distinct !{!64, !65, !"_ZN7__spirvL21initLocalInvocationIdILi2EN4sycl3_V12idILi2EEEEET0_v: %agg.result"}
!65 = distinct !{!65, !"_ZN7__spirvL21initLocalInvocationIdILi2EN4sycl3_V12idILi2EEEEET0_v"}
!66 = !{!67}
!67 = distinct !{!67, !68, !"_ZN4sycl3_V13ext6oneapi12experimental6matrix16joint_matrix_madINS0_9sub_groupENS2_8bfloat16ES7_fLm8ELm16ELm16ELNS4_6layoutE0ELS8_2EEENS4_12joint_matrixIT_T2_LNS4_3useE2EXT3_EXT5_ELS8_3EEESA_RNS9_ISA_T0_LSC_0EXT3_EXT4_EXT6_EEERNS9_ISA_T1_LSC_1EXT4_EXT5_EXT7_EEERSD_: %agg.result"}
!68 = distinct !{!68, !"_ZN4sycl3_V13ext6oneapi12experimental6matrix16joint_matrix_madINS0_9sub_groupENS2_8bfloat16ES7_fLm8ELm16ELm16ELNS4_6layoutE0ELS8_2EEENS4_12joint_matrixIT_T2_LNS4_3useE2EXT3_EXT5_ELS8_3EEESA_RNS9_ISA_T0_LSC_0EXT3_EXT4_EXT6_EEERNS9_ISA_T1_LSC_1EXT4_EXT5_EXT7_EEERSD_"}
!69 = distinct !{!69, !70}
!70 = !{!"llvm.loop.mustprogress"}
