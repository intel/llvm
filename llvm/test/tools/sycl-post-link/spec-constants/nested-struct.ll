; This test demonstrates that multiple padding elements can be
; inserted in the spec constant metadata

; RUN: sycl-post-link --spec-const=native -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll
; RUN: sycl-post-link -debug-only=SpecConst -spec-const=native < %s 2>&1 | FileCheck %s --check-prefix=CHECK-LOG 

; CHECK: %[[#SCV1:]] = call i8 @_Z20__spirv_SpecConstantia(i32 [[#SCID1:]], i8 120)
; CHECK: %[[#SCV2:]] = call i8 @_Z20__spirv_SpecConstantia(i32 [[#SCID2:]], i8 121)
; CHECK: %[[#SCV3:]] = call i32 @_Z20__spirv_SpecConstantii(i32 [[#SCID3:]], i32 122)
; CHECK: %[[#SCV4:]] = call i8 @_Z20__spirv_SpecConstantia(i32 [[#SCID4:]], i8 97)
; CHECK: %[[#SCV5:]] = call %struct.anon @_Z29__spirv_SpecConstantCompositeaia_Rstruct.anon(i8 %[[#SCV2:]], i32 %[[#SCV3:]], i8 %[[#SCV4:]])
; CHECK: %[[#SCV6:]] = call i8 @_Z20__spirv_SpecConstantia(i32 [[#SCID5:]], i8 98)
; CHECK: call %struct.user_defined_type3 @_Z29__spirv_SpecConstantCompositeastruct.anona_Rstruct.user_defined_type3(i8 %[[#SCV1:]], %struct.anon %[[#SCV5:]], i8 %[[#SCV6:]])

; CHECK: !sycl.specialization-constants = !{![[#SC:]]}
; CHECK: ![[#SC]] = !{!"uid0a28d8a0a23067ab____ZL8spec_id3",
; CHECK-SAME: i32 [[#SCID1:]], i32 0, i32 1,
; CHECK-SAME: i32 [[#SCID2:]], i32 4, i32 1,
; CHECK-SAME: i32 [[#SCID3:]], i32 8, i32 4,
; CHECK-SAME: i32 [[#SCID4:]], i32 12, i32 1,
; CHECK-SAME: i32 -1, i32 13, i32 3,
; CHECK-SAME: i32 [[#SCID5:]], i32 16, i32 1,
; CHECK-SAME: i32 -1, i32 17, i32 3}
; CHECK-LOG: sycl.specialization-constants
; CHECK-LOG:[[UNIQUE_PREFIX:[0-9a-zA-Z]+]]={0, 0, 1}
; CHECK-LOG:[[UNIQUE_PREFIX]]={1, 4, 1}
; CHECK-LOG:[[UNIQUE_PREFIX]]={2, 8, 4}
; CHECK-LOG:[[UNIQUE_PREFIX]]={3, 12, 1}
; CHECK-LOG:[[UNIQUE_PREFIX]]={4294967295, 13, 3}
; CHECK-LOG:[[UNIQUE_PREFIX]]={4, 16, 1}
; CHECK-LOG:[[UNIQUE_PREFIX]]={4294967295, 17, 3}
; CHECK-LOG: sycl.specialization-constants-default-values
; CHECK-LOG:{0, 1, 120}
; CHECK-LOG:{4, 1, 121}
; CHECK-LOG:{8, 4, 122}
; CHECK-LOG:{12, 1, 97}
; CHECK-LOG:{16, 1, 98}

; ModuleID = '..\sycl\test-e2e\SpecConstants\2020\nested-non-packed-struct.cpp'
source_filename = "..\\sycl\\test-e2e\\SpecConstants\\2020\\nested-non-packed-struct.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::specialization_id.24" = type { %struct.user_defined_type3 }
%struct.user_defined_type3 = type { i8, %struct.anon, i8 }
%struct.anon = type { i8, i32, i8 }
%"class.sycl::_V1::id" = type { %"class.sycl::_V1::detail::array" }
%"class.sycl::_V1::detail::array" = type { [1 x i64] }

@__usid_str.2 = private unnamed_addr constant [35 x i8] c"uid0a28d8a0a23067ab____ZL8spec_id3\00", align 1
@_ZL8spec_id3 = internal addrspace(1) constant %"class.sycl::_V1::specialization_id.24" { %struct.user_defined_type3 { i8 120, %struct.anon { i8 121, i32 122, i8 97 }, i8 98 } }, align 4

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE2_clES2_EUlNS0_14kernel_handlerEE_(ptr addrspace(1) noundef align 4 %_arg_acc, ptr noundef byval(%"class.sycl::_V1::id") align 8 %_arg_acc3) local_unnamed_addr #0 !srcloc !78 !kernel_arg_buffer_location !63 !kernel_arg_runtime_aligned !64 !kernel_arg_exclusive_ptr !64 !sycl_fixed_targets !65 !sycl_kernel_omit_args !66 {
entry:
  %ref.tmp.i = alloca %struct.user_defined_type3, align 4
  %0 = load i64, ptr %_arg_acc3, align 8
  %add.ptr.i = getelementptr inbounds %struct.user_defined_type3, ptr addrspace(1) %_arg_acc, i64 %0
  %ref.tmp.ascast.i = addrspacecast ptr %ref.tmp.i to ptr addrspace(4)
  call spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI18user_defined_type3ET_PKcPKvS5_(ptr addrspace(4) sret(%struct.user_defined_type3) align 4 %ref.tmp.ascast.i, ptr addrspace(4) noundef addrspacecast (ptr @__usid_str.2 to ptr addrspace(4)), ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @_ZL8spec_id3 to ptr addrspace(4)), ptr addrspace(4) noundef null) #5
  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align 4 %add.ptr.i, ptr align 4 %ref.tmp.i, i64 20, i1 false), !tbaa.struct !79
  ret void
}

; Function Attrs: convergent nounwind
declare dso_local spir_func void @_Z40__sycl_getComposite2020SpecConstantValueI18user_defined_type3ET_PKcPKvS5_(ptr addrspace(4) sret(%struct.user_defined_type3) align 4, ptr addrspace(4) noundef, ptr addrspace(4) noundef, ptr addrspace(4) noundef) local_unnamed_addr #2

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #3

attributes #0 = { convergent norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="..\\sycl\\test-e2e\\SpecConstants\\2020\\nested-non-packed-struct.cpp" "sycl-optlevel"="2" "sycl-single-task" "uniform-work-group-size"="true" }
attributes #2 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nounwind }
attributes #5 = { convergent nounwind }

!llvm.dependent-libraries = !{!0}
!llvm.module.flags = !{!1, !2}
!opencl.spir.version = !{!3}
!spirv.Source = !{!4}
!sycl_aspects = !{!5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60}
!llvm.ident = !{!61}

!0 = !{!"libcpmt"}
!1 = !{i32 1, !"wchar_size", i32 2}
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
!55 = !{!"ext_oneapi_non_uniform_groups", i32 54}
!56 = !{!"int64_base_atomics", i32 7}
!57 = !{!"int64_extended_atomics", i32 8}
!58 = !{!"usm_system_allocator", i32 17}
!59 = !{!"usm_restricted_shared_allocations", i32 16}
!60 = !{!"host", i32 0}
!61 = !{!"clang version 18.0.0 (https://github.com/intel/llvm.git c92b6b0c266b6a0d5bca1d61a63f06e2bce37904)"}
!62 = !{i32 2091}
!63 = !{i32 -1, i32 -1}
!64 = !{i1 true, i1 false}
!65 = !{}
!66 = !{i1 false, i1 true, i1 true, i1 false, i1 true}
!67 = !{i64 0, i64 4, !68, i64 4, i64 1, !72, i64 8, i64 4, !73}
!68 = !{!69, !69, i64 0}
!69 = !{!"float", !70, i64 0}
!70 = !{!"omnipotent char", !71, i64 0}
!71 = !{!"Simple C++ TBAA"}
!72 = !{!70, !70, i64 0}
!73 = !{!74, !74, i64 0}
!74 = !{!"int", !70, i64 0}
!75 = !{i32 2451}
!76 = !{i64 0, i64 4, !68, i64 32, i64 4, !68, i64 36, i64 1, !72, i64 40, i64 4, !73, i64 64, i64 4, !73, i64 68, i64 1, !72}
!77 = !{i32 2882}
!78 = !{i32 3230}
!79 = !{i64 0, i64 1, !72, i64 4, i64 1, !72, i64 8, i64 4, !73, i64 12, i64 1, !72, i64 16, i64 1, !72}
