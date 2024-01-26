; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -S | FileCheck %s

; ModuleID = 'spir.cpp'
source_filename = "spir.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::range" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E10FillBuffer = comdat any

@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
; CHECK: __AsanShadowMemoryGlobalStart

; Function Attrs: convergent mustprogress norecurse nounwind sanitize_address uwtable
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E10FillBuffer(ptr addrspace(1) noundef align 8 %_arg_Sum, ptr addrspace(1) noundef align 8 %_arg_Accessor, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_Accessor1, ptr noundef byval(%"class.sycl::_V1::range") align 8 %_arg_Accessor2, ptr noundef byval(%"class.sycl::_V1::id") align 8 %_arg_Accessor3) local_unnamed_addr #0 comdat !srcloc !65 !kernel_arg_buffer_location !66 !kernel_arg_runtime_aligned !67 !kernel_arg_exclusive_ptr !67 !sycl_fixed_targets !68 {
entry:
  %0 = load i64, ptr %_arg_Accessor3, align 8
  ; CHECK: __asan_load8
  %1 = load i64, ptr addrspace(1) @__spirv_BuiltInLocalInvocationId, align 32, !noalias !68
  %call.i10 = tail call spir_func ptr addrspace(3) @__sycl_allocateLocalMemory(i64 noundef 16, i64 noundef 4) #5, !noalias !69
  ; CHECK: __asan_set_shadow_local_memory
  %cmp.i = icmp eq i64 %1, 0
  br i1 %cmp.i, label %if.then.i, label %_ZN4sycl3_V13ext6oneapi18group_local_memoryIA4_iNS0_5groupILi1EEEJEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS8_LNS0_6access13address_spaceE3ELNSB_9decoratedE2EEEE4typeES9_DpOT1_.exit

if.then.i:                                        ; preds = %entry
  call void @llvm.memset.p3.i64(ptr addrspace(3) align 4 %call.i10, i8 0, i64 16, i1 false), !noalias !69
  br label %_ZN4sycl3_V13ext6oneapi18group_local_memoryIA4_iNS0_5groupILi1EEEJEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS8_LNS0_6access13address_spaceE3ELNSB_9decoratedE2EEEE4typeES9_DpOT1_.exit

_ZN4sycl3_V13ext6oneapi18group_local_memoryIA4_iNS0_5groupILi1EEEJEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS8_LNS0_6access13address_spaceE3ELNSB_9decoratedE2EEEE4typeES9_DpOT1_.exit: ; preds = %entry, %if.then.i
  %add.ptr.i = getelementptr inbounds i64, ptr addrspace(1) %_arg_Accessor, i64 %0
  tail call spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef 2, i32 noundef 2, i32 noundef 272) #5, !noalias !69
  %cmp.i14 = icmp ult i64 %1, 2147483648
  tail call void @llvm.assume(i1 %cmp.i14)
  %arrayidx.i16 = getelementptr inbounds i64, ptr addrspace(1) %add.ptr.i, i64 %1
  %2 = load i64, ptr addrspace(1) %arrayidx.i16, align 8, !tbaa !72
  ; CHECK: __asan_load8
  %arrayidx.i = getelementptr inbounds [4 x i32], ptr addrspace(3) %call.i10, i64 0, i64 %1
  %3 = load i32, ptr addrspace(3) %arrayidx.i, align 4, !tbaa !76
  ; CHECK: __asan_load4
  %conv.i = sext i32 %3 to i64
  %add.i = add i64 %2, %conv.i
  %4 = load i64, ptr addrspace(1) %_arg_Sum, align 8, !tbaa !72
  ; CHECK: __asan_load8
  %add5.i = add i64 %4, %add.i
  store i64 %add5.i, ptr addrspace(1) %_arg_Sum, align 8, !tbaa !72
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p4.i64(ptr addrspace(4) nocapture writeonly %0, i8 %1, i64 %2, i1 immarg %3) #1

; Function Attrs: convergent nounwind
declare dso_local spir_func ptr addrspace(3) @__sycl_allocateLocalMemory(i64 noundef %0, i64 noundef %1) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z22__spirv_ControlBarrierjjj(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef %0) #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p3.i64(ptr addrspace(3) nocapture writeonly %0, i8 %1, i64 %2, i1 immarg %3) #4

attributes #0 = { convergent mustprogress norecurse nounwind sanitize_address uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test.cpp" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #5 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2}
!opencl.spir.version = !{!3}
!spirv.Source = !{!4}
!sycl_aspects = !{!5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63}
!llvm.ident = !{!64}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 1, i32 2}
!4 = !{i32 4, i32 100000}
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
!43 = !{!"ext_oneapi_bindless_images", i32 42}
!44 = !{!"ext_oneapi_bindless_images_shared_usm", i32 43}
!45 = !{!"ext_oneapi_bindless_images_1d_usm", i32 44}
!46 = !{!"ext_oneapi_bindless_images_2d_usm", i32 45}
!47 = !{!"ext_oneapi_interop_memory_import", i32 46}
!48 = !{!"ext_oneapi_interop_memory_export", i32 47}
!49 = !{!"ext_oneapi_interop_semaphore_import", i32 48}
!50 = !{!"ext_oneapi_interop_semaphore_export", i32 49}
!51 = !{!"ext_oneapi_mipmap", i32 50}
!52 = !{!"ext_oneapi_mipmap_anisotropy", i32 51}
!53 = !{!"ext_oneapi_mipmap_level_reference", i32 52}
!54 = !{!"ext_intel_esimd", i32 53}
!55 = !{!"ext_oneapi_ballot_group", i32 54}
!56 = !{!"ext_oneapi_fixed_size_group", i32 55}
!57 = !{!"ext_oneapi_opportunistic_group", i32 56}
!58 = !{!"ext_oneapi_tangle_group", i32 57}
!59 = !{!"int64_base_atomics", i32 7}
!60 = !{!"int64_extended_atomics", i32 8}
!61 = !{!"usm_system_allocator", i32 17}
!62 = !{!"usm_restricted_shared_allocations", i32 16}
!63 = !{!"host", i32 0}
!64 = !{!"clang version 18.0.0git (https://github.com/intel/llvm.git caecf6b928648a83c8ceb84988231cb246c4365e)"}
!65 = !{i32 419}
!66 = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}
!67 = !{i1 false, i1 true, i1 false, i1 false, i1 false}
!68 = !{}
!69 = !{!70}
!70 = distinct !{!70, !71, !"_ZN4sycl3_V13ext6oneapi18group_local_memoryIA4_iNS0_5groupILi1EEEJEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS8_LNS0_6access13address_spaceE3ELNSB_9decoratedE2EEEE4typeES9_DpOT1_: %agg.result"}
!71 = distinct !{!71, !"_ZN4sycl3_V13ext6oneapi18group_local_memoryIA4_iNS0_5groupILi1EEEJEEENSt9enable_ifIXaasr3stdE27is_trivially_destructible_vIT_Esr4sycl6detail8is_groupIT0_EE5valueENS0_9multi_ptrIS8_LNS0_6access13address_spaceE3ELNSB_9decoratedE2EEEE4typeES9_DpOT1_"}
!72 = !{!73, !73, i64 0}
!73 = !{!"long", !74, i64 0}
!74 = !{!"omnipotent char", !75, i64 0}
!75 = !{!"Simple C++ TBAA"}
!76 = !{!77, !77, i64 0}
!77 = !{!"int", !74, i64 0}
