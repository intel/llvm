; This checks that sycl-post-link can accept multiple -o options, 
; with some of the -o options being composed of a (target, filename) pair,
; and that the output tables from inputs with target info have the modules
; that are not compatible with that target filtered out.

; RUN: sycl-post-link %s -symbols -split=auto \
; RUN: -o %t.table \
; RUN: -o intel_gpu_pvc,%t-pvc.table \
; RUN: -o intel_gpu_tgllp,%t-tgllp.table \
; RUN: -o intel_gpu_cfl,%t-cfl.table \
; RUN: -o unrecognized_target,%t-unrecognized.table

; RUN: FileCheck %s -input-file=%t_0.sym -check-prefix=CHECK-DOUBLE
; RUN: FileCheck %s -input-file=%t_1.sym -check-prefix=CHECK-SG8
; RUN: FileCheck %s -input-file=%t_2.sym -check-prefix=CHECK-SG64
; RUN: FileCheck %s -input-file=%t_3.sym -check-prefix=CHECK-SG32
; RUN: FileCheck %s -input-file=%t_4.sym -check-prefix=CHECK-FLOAT
; RUN: FileCheck %s -input-file=%t_5.sym -check-prefix=CHECK-SG16

; RUN: FileCheck %s -input-file=%t.table -check-prefix=CHECK-ALL
; RUN: FileCheck %s -input-file=%t-unrecognized.table -check-prefix=CHECK-ALL
; RUN: FileCheck %s -input-file=%t-pvc.table -check-prefix=CHECK-PVC
; RUN: FileCheck %s -input-file=%t-tgllp.table -check-prefix=CHECK-TGLLP
; RUN: FileCheck %s -input-file=%t-cfl.table -check-prefix=CHECK-CFL

; CHECK-DOUBLE: double_kernel
; CHECK-FLOAT: float_kernel
; CHECK-SG8: reqd_sub_group_size_kernel_8
; CHECK-SG16: reqd_sub_group_size_kernel_16
; CHECK-SG32: reqd_sub_group_size_kernel_32
; CHECK-SG64: reqd_sub_group_size_kernel_64

; An output without a target will have no filtering performed on the output table.
; Additionally, an unrecognized target will perform the same.
; CHECK-ALL:      _0.sym
; CHECK-ALL-NEXT: _1.sym
; CHECK-ALL-NEXT: _2.sym
; CHECK-ALL-NEXT: _3.sym
; CHECK-ALL-NEXT: _4.sym
; CHECK-ALL-NEXT: _5.sym
; CHECK-ALL-EMPTY:

; PVC does not support sg8 (=1) or sg64 (=2) 
; CHECK-PVC:      _0.sym
; CHECK-PVC-NEXT: _3.sym
; CHECK-PVC-NEXT: _4.sym
; CHECK-PVC-NEXT: _5.sym
; CHECK-PVC-EMPTY:

; TGLLP does not support fp64 (=0) or sg64 (=2)
; CHECK-TGLLP:      _1.sym
; CHECK-TGLLP-NEXT: _3.sym
; CHECK-TGLLP-NEXT: _4.sym
; CHECK-TGLLP-NEXT: _5.sym
; CHECK-TGLLP-EMPTY:

; CFL does not support sg64 (=2)
; CHECK-CFL:      _0.sym
; CHECK-CFL-NEXT: _1.sym
; CHECK-CFL-NEXT: _3.sym
; CHECK-CFL-NEXT: _4.sym
; CHECK-CFL-NEXT: _5.sym
; CHECK-CFL-EMPTY:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @double_kernel(ptr addrspace(1) noundef align 8 %_arg_out) local_unnamed_addr #0 !srcloc !65 !kernel_arg_buffer_location !66 !sycl_used_aspects !67 !sycl_fixed_targets !68 !sycl_kernel_omit_args !69 {
entry:
  %0 = load double, ptr addrspace(1) %_arg_out, align 8, !tbaa !70
  %mul.i = fmul double %0, 2.000000e-01
  store double %mul.i, ptr addrspace(1) %_arg_out, align 8, !tbaa !70
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @float_kernel(ptr addrspace(1) noundef align 4 %_arg_out) local_unnamed_addr #0 !srcloc !74 !kernel_arg_buffer_location !66 !sycl_fixed_targets !68 !sycl_kernel_omit_args !69 {
entry:
  %0 = load float, ptr addrspace(1) %_arg_out, align 4, !tbaa !75
  %mul.i = fmul float %0, 0x3FC99999A0000000
  store float %mul.i, ptr addrspace(1) %_arg_out, align 4, !tbaa !75
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @reqd_sub_group_size_kernel_8() local_unnamed_addr #0 !srcloc !77 !kernel_arg_buffer_location !68 !intel_reqd_sub_group_size !78 !sycl_fixed_targets !68 !sycl_kernel_omit_args !68 {
entry:
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @reqd_sub_group_size_kernel_16() local_unnamed_addr #0 !srcloc !77 !kernel_arg_buffer_location !68 !intel_reqd_sub_group_size !79 !sycl_fixed_targets !68 !sycl_kernel_omit_args !68 {
entry:
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @reqd_sub_group_size_kernel_32() local_unnamed_addr #0 !srcloc !77 !kernel_arg_buffer_location !68 !intel_reqd_sub_group_size !80 !sycl_fixed_targets !68 !sycl_kernel_omit_args !68 {
entry:
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @reqd_sub_group_size_kernel_64() local_unnamed_addr #0 !srcloc !77 !kernel_arg_buffer_location !68 !intel_reqd_sub_group_size !81 !sycl_fixed_targets !68 !sycl_kernel_omit_args !68 {
entry:
  ret void
}

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(ptr addrspace(2), ...)

attributes #0 = { mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="double.cpp" "sycl-optlevel"="3" "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!sycl_aspects = !{!4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63}
!llvm.ident = !{!64}

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
!42 = !{!"ext_oneapi_bindless_images", i32 42}
!43 = !{!"ext_oneapi_bindless_images_shared_usm", i32 43}
!44 = !{!"ext_oneapi_bindless_images_1d_usm", i32 44}
!45 = !{!"ext_oneapi_bindless_images_2d_usm", i32 45}
!46 = !{!"ext_oneapi_interop_memory_import", i32 46}
!47 = !{!"ext_oneapi_interop_memory_export", i32 47}
!48 = !{!"ext_oneapi_interop_semaphore_import", i32 48}
!49 = !{!"ext_oneapi_interop_semaphore_export", i32 49}
!50 = !{!"ext_oneapi_mipmap", i32 50}
!51 = !{!"ext_oneapi_mipmap_anisotropy", i32 51}
!52 = !{!"ext_oneapi_mipmap_level_reference", i32 52}
!53 = !{!"ext_intel_esimd", i32 53}
!54 = !{!"ext_oneapi_ballot_group", i32 54}
!55 = !{!"ext_oneapi_fixed_size_group", i32 55}
!56 = !{!"ext_oneapi_opportunistic_group", i32 56}
!57 = !{!"ext_oneapi_tangle_group", i32 57}
!58 = !{!"ext_intel_matrix", i32 58}
!59 = !{!"int64_base_atomics", i32 7}
!60 = !{!"int64_extended_atomics", i32 8}
!61 = !{!"usm_system_allocator", i32 17}
!62 = !{!"usm_restricted_shared_allocations", i32 16}
!63 = !{!"host", i32 0}
!64 = !{!"clang version 19.0.0git (/ws/llvm/clang a7f3a637bdd6299831f903bbed9e8d069fea5c86)"}
!65 = !{i32 233}
!66 = !{i32 -1}
!67 = !{i32 6}
!68 = !{}
!69 = !{i1 false}
!70 = !{!71, !71, i64 0}
!71 = !{!"double", !72, i64 0}
!72 = !{!"omnipotent char", !73, i64 0}
!73 = !{!"Simple C++ TBAA"}
!74 = !{i32 364}
!75 = !{!76, !76, i64 0}
!76 = !{!"float", !72, i64 0}
!77 = !{i32 529}
!78 = !{i32 8}
!79 = !{i32 16}
!80 = !{i32 32}
!81 = !{i32 64}
