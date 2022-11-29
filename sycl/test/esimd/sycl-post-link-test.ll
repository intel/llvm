; RUN: sycl-post-link -split-esimd -lower-esimd -O2 -S %t -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll
; This test checks that IR code below can be successfully processed by
; sycl-post-link. In this IR no extractelement instruction and no casting are used

; ModuleID = 'sycl-post-link-test.cpp'
source_filename = "sycl-post-link-test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::intel::esimd::simd" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl" = type { <8 x float> }
%"class.sycl::_V1::ext::intel::esimd::simd.0" = type { %"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.1" }
%"class.sycl::_V1::ext::intel::esimd::detail::simd_obj_impl.1" = type { <8 x i32> }

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: convergent norecurse
define dso_local spir_func void @vmult2(%"class.sycl::_V1::ext::intel::esimd::simd"* noundef %a) local_unnamed_addr #0 !srcloc !47 !sycl_explicit_simd !45 !intel_reqd_sub_group_size !48 !sycl_fixed_targets !45 {
entry:
  %Res.i.i.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %Res.i.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd", align 32
  %ref.tmp.i = alloca %"class.sycl::_V1::ext::intel::esimd::simd.0", align 32
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32
  %conv = trunc i64 %0 to i32
  %1 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd.0"* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %1) #3
  %splat.splatinsert.i.i.i = insertelement <8 x i32> poison, i32 %conv, i64 0
  %splat.splat.i.i.i = shufflevector <8 x i32> %splat.splatinsert.i.i.i, <8 x i32> poison, <8 x i32> zeroinitializer
  %M_data.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd.0", %"class.sycl::_V1::ext::intel::esimd::simd.0"* %ref.tmp.i, i64 0, i32 0, i32 0
  %2 = addrspacecast <8 x i32>* %M_data.i.i.i to <8 x i32> addrspace(4)*
  store <8 x i32> %splat.splat.i.i.i, <8 x i32>* %M_data.i.i.i, align 32, !tbaa !49
  %3 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %Res.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %3) #3
  %4 = bitcast %"class.sycl::_V1::ext::intel::esimd::simd"* %Res.i.i.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %4) #3, !noalias !52
  %M_data.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %a, i64 0, i32 0, i32 0
  %5 = addrspacecast <8 x float>* %M_data.i.i.i.i to <8 x float> addrspace(4)*
  %6 = addrspacecast <8 x float>* %M_data.i.i.i.i to <8 x float> addrspace(4)*
  %call.i.i.i.i = tail call spir_func noundef <8 x float> @_Z13__esimd_vloadIfLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x float> addrspace(4)* noundef %6) #4, !noalias !52
  %call.i6.i.i.i = call spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef %2) #4, !noalias !52
  %conv.i.i.i.i = sitofp <8 x i32> %call.i6.i.i.i to <8 x float>
  %mul.i.i.i.i.i.i = fmul <8 x float> %call.i.i.i.i, %conv.i.i.i.i
  %M_data.i.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %Res.i.i.i, i64 0, i32 0, i32 0
  %7 = addrspacecast <8 x float>* %M_data.i.i.i.i.i.i to <8 x float> addrspace(4)*
  %8 = addrspacecast <8 x float>* %M_data.i.i.i.i.i.i to <8 x float> addrspace(4)*
  call spir_func void @_Z14__esimd_vstoreIfLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x float> addrspace(4)* noundef %7, <8 x float> noundef %mul.i.i.i.i.i.i) #4, !noalias !52
  %call.i.i.i.i.i.i = call spir_func noundef <8 x float> @_Z13__esimd_vloadIfLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x float> addrspace(4)* noundef %8) #4, !noalias !52
  %M_data.i2.i.i.i.i.i = getelementptr inbounds %"class.sycl::_V1::ext::intel::esimd::simd", %"class.sycl::_V1::ext::intel::esimd::simd"* %Res.i.i, i64 0, i32 0, i32 0
  %9 = addrspacecast <8 x float>* %M_data.i2.i.i.i.i.i to <8 x float> addrspace(4)*
  %10 = addrspacecast <8 x float>* %M_data.i2.i.i.i.i.i to <8 x float> addrspace(4)*
  call spir_func void @_Z14__esimd_vstoreIfLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x float> addrspace(4)* noundef %9, <8 x float> noundef %call.i.i.i.i.i.i) #4
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %4) #3, !noalias !52
  %call.i.i.i = call spir_func noundef <8 x float> @_Z13__esimd_vloadIfLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x float> addrspace(4)* noundef %10) #4
  call spir_func void @_Z14__esimd_vstoreIfLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x float> addrspace(4)* noundef %5, <8 x float> noundef %call.i.i.i) #4
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %3) #3
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %1) #3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: convergent
declare dso_local spir_func noundef <8 x i32> @_Z13__esimd_vloadIiLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x i32> addrspace(4)* noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func void @_Z14__esimd_vstoreIfLi8EEvPN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeES9_(<8 x float> addrspace(4)* noundef, <8 x float> noundef) local_unnamed_addr #2

; Function Attrs: convergent
declare dso_local spir_func noundef <8 x float> @_Z13__esimd_vloadIfLi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT_XT0_EE4typeEPKS9_(<8 x float> addrspace(4)* noundef) local_unnamed_addr #2

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...)

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="256" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="sycl-post-link-test.cpp" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nounwind }
attributes #4 = { convergent }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!sycl_aspects = !{!4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44}
!opencl.compiler.options = !{!45}
!llvm.ident = !{!46}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 0, i32 100000}
!4 = !{!"host", i32 0}
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
!18 = !{!"usm_restricted_shared_allocations", i32 16}
!19 = !{!"usm_system_allocations", i32 17}
!20 = !{!"ext_intel_pci_address", i32 18}
!21 = !{!"ext_intel_gpu_eu_count", i32 19}
!22 = !{!"ext_intel_gpu_eu_simd_width", i32 20}
!23 = !{!"ext_intel_gpu_slices", i32 21}
!24 = !{!"ext_intel_gpu_subslices_per_slice", i32 22}
!25 = !{!"ext_intel_gpu_eu_count_per_subslice", i32 23}
!26 = !{!"ext_intel_max_mem_bandwidth", i32 24}
!27 = !{!"ext_intel_mem_channel", i32 25}
!28 = !{!"usm_atomic_host_allocations", i32 26}
!29 = !{!"usm_atomic_shared_allocations", i32 27}
!30 = !{!"atomic64", i32 28}
!31 = !{!"ext_intel_device_info_uuid", i32 29}
!32 = !{!"ext_oneapi_srgb", i32 30}
!33 = !{!"ext_oneapi_native_assert", i32 31}
!34 = !{!"host_debuggable", i32 32}
!35 = !{!"ext_intel_gpu_hw_threads_per_eu", i32 33}
!36 = !{!"ext_oneapi_cuda_async_barrier", i32 34}
!37 = !{!"ext_oneapi_bfloat16", i32 35}
!38 = !{!"ext_intel_free_memory", i32 36}
!39 = !{!"ext_intel_device_id", i32 37}
!40 = !{!"ext_intel_memory_clock_rate", i32 38}
!41 = !{!"ext_intel_memory_bus_width", i32 39}
!42 = !{!"int64_base_atomics", i32 7}
!43 = !{!"int64_extended_atomics", i32 8}
!44 = !{!"usm_system_allocator", i32 17}
!45 = !{}
!46 = !{!"Intel(R) oneAPI DPC++/C++ Compiler 2023.1.0 (2023.x.0.YYYYMMDD)"}
!47 = !{i32 606}
!48 = !{i32 1}
!49 = !{!50, !50, i64 0}
!50 = !{!"omnipotent char", !51, i64 0}
!51 = !{!"Simple C++ TBAA"}
!52 = !{!53}
!53 = distinct !{!53, !54, !"_ZN4sycl3_V13ext5intel5esimd6detailmlIfiLi8ENS3_4simdENS6_IfLi8EEEvEEDaRKNS4_13simd_obj_implINS4_19element_type_traitsIT_vE4RawTEXT1_ET2_ISA_XT1_EEvEERKNS8_INS9_IT0_vE4RawTEXT1_ESD_ISI_XT1_EEvEE: %agg.result"}
!54 = distinct !{!54, !"_ZN4sycl3_V13ext5intel5esimd6detailmlIfiLi8ENS3_4simdENS6_IfLi8EEEvEEDaRKNS4_13simd_obj_implINS4_19element_type_traitsIT_vE4RawTEXT1_ET2_ISA_XT1_EEvEERKNS8_INS9_IT0_vE4RawTEXT1_ESD_ISI_XT1_EEvEE"}


; CHECK: define dso_local spir_func void @vmult2
; CHECK:   call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK:   call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK:   call i32 @llvm.genx.group.id.x()
; CHECK:   ret void
; CHECK: }