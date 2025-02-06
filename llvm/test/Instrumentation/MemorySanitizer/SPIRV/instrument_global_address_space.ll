; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -S | FileCheck %s

; ModuleID = 'check_call.cpp'
source_filename = "check_call.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel = comdat any

; CHECK: @__MsanKernelMetadata = appending dso_local local_unnamed_addr addrspace(1) global
; CHECK-SAME: [[ATTR0:#[0-9]+]]

; Function Attrs: mustprogress norecurse nounwind sanitize_memory uwtable
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel(ptr addrspace(1) noundef align 4 %_arg_array) local_unnamed_addr #0 comdat !srcloc !85 !kernel_arg_buffer_location !86 !sycl_fixed_targets !87 {
; CHECK-LABEL: @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E8MyKernel
entry:
; CHECK-NOT: @__msan_param_tls
  %0 = load i32, ptr addrspace(1) %_arg_array, align 4, !tbaa !88
  %arrayidx3.i = getelementptr inbounds i8, ptr addrspace(1) %_arg_array, i64 4
; CHECK: @__msan_get_shadow
  %1 = load i32, ptr addrspace(1) %arrayidx3.i, align 4, !tbaa !88
  %conv.i = sext i32 %1 to i64
  %call.i = tail call spir_func noundef i64 @_Z3fooix(i32 noundef %0, i64 noundef %conv.i) #2
  %conv4.i = trunc i64 %call.i to i32
  store i32 %conv4.i, ptr addrspace(1) %_arg_array, align 4, !tbaa !88
  ret void
}

; Function Attrs: mustprogress noinline norecurse nounwind sanitize_memory uwtable
define linkonce_odr dso_local spir_func noundef i64 @_Z3fooix(i32 noundef %data1, i64 noundef %data2) local_unnamed_addr #1 !srcloc !92 {
; CHECK-LABEL: @_Z3fooix
entry:
  %conv = sext i32 %data1 to i64
  %add = add nsw i64 %data2, %conv
  ret i64 %add
}

; CHECK: attributes [[ATTR0]]
; CHECK-SAME: "sycl-device-global-size"="16" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="_Z20__MsanKernelMetadata"

attributes #0 = { mustprogress norecurse nounwind sanitize_memory uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="check_call.cpp" "sycl-single-task" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress noinline norecurse nounwind sanitize_memory uwtable "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.spir.version = !{!4}
!spirv.Source = !{!5}
!sycl_aspects = !{!6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83}
!llvm.ident = !{!84}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"sycl-device", i32 1}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{i32 1, i32 2}
!5 = !{i32 4, i32 100000}
!6 = !{!"cpu", i32 1}
!7 = !{!"gpu", i32 2}
!8 = !{!"accelerator", i32 3}
!9 = !{!"custom", i32 4}
!10 = !{!"fp16", i32 5}
!11 = !{!"fp64", i32 6}
!12 = !{!"image", i32 9}
!13 = !{!"online_compiler", i32 10}
!14 = !{!"online_linker", i32 11}
!15 = !{!"queue_profiling", i32 12}
!16 = !{!"usm_device_allocations", i32 13}
!17 = !{!"usm_host_allocations", i32 14}
!18 = !{!"usm_shared_allocations", i32 15}
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
!47 = !{!"ext_oneapi_external_memory_import", i32 46}
!48 = !{!"ext_oneapi_external_semaphore_import", i32 48}
!49 = !{!"ext_oneapi_mipmap", i32 50}
!50 = !{!"ext_oneapi_mipmap_anisotropy", i32 51}
!51 = !{!"ext_oneapi_mipmap_level_reference", i32 52}
!52 = !{!"ext_intel_esimd", i32 53}
!53 = !{!"ext_oneapi_ballot_group", i32 54}
!54 = !{!"ext_oneapi_fixed_size_group", i32 55}
!55 = !{!"ext_oneapi_opportunistic_group", i32 56}
!56 = !{!"ext_oneapi_tangle_group", i32 57}
!57 = !{!"ext_intel_matrix", i32 58}
!58 = !{!"ext_oneapi_is_composite", i32 59}
!59 = !{!"ext_oneapi_is_component", i32 60}
!60 = !{!"ext_oneapi_graph", i32 61}
!61 = !{!"ext_intel_fpga_task_sequence", i32 62}
!62 = !{!"ext_oneapi_limited_graph", i32 63}
!63 = !{!"ext_oneapi_private_alloca", i32 64}
!64 = !{!"ext_oneapi_cubemap", i32 65}
!65 = !{!"ext_oneapi_cubemap_seamless_filtering", i32 66}
!66 = !{!"ext_oneapi_bindless_sampled_image_fetch_1d_usm", i32 67}
!67 = !{!"ext_oneapi_bindless_sampled_image_fetch_1d", i32 68}
!68 = !{!"ext_oneapi_bindless_sampled_image_fetch_2d_usm", i32 69}
!69 = !{!"ext_oneapi_bindless_sampled_image_fetch_2d", i32 70}
!70 = !{!"ext_oneapi_bindless_sampled_image_fetch_3d", i32 72}
!71 = !{!"ext_oneapi_queue_profiling_tag", i32 73}
!72 = !{!"ext_oneapi_virtual_mem", i32 74}
!73 = !{!"ext_oneapi_cuda_cluster_group", i32 75}
!74 = !{!"ext_oneapi_image_array", i32 76}
!75 = !{!"ext_oneapi_unique_addressing_per_dim", i32 77}
!76 = !{!"ext_oneapi_bindless_images_sample_1d_usm", i32 78}
!77 = !{!"ext_oneapi_bindless_images_sample_2d_usm", i32 79}
!78 = !{!"ext_oneapi_atomic16", i32 80}
!79 = !{!"ext_oneapi_virtual_functions", i32 81}
!80 = !{!"host", i32 0}
!81 = !{!"int64_base_atomics", i32 7}
!82 = !{!"int64_extended_atomics", i32 8}
!83 = !{!"usm_restricted_shared_allocations", i32 16}
!84 = !{!"clang version 20.0.0git (https://github.com/intel/llvm.git 7384106e6410c6f038b2a9d6367a32b55278c638)"}
!85 = !{i32 563}
!86 = !{i32 -1}
!87 = !{}
!88 = !{!89, !89, i64 0}
!89 = !{!"int", !90, i64 0}
!90 = !{!"omnipotent char", !91, i64 0}
!91 = !{!"Simple C++ TBAA"}
!92 = !{i32 345}
