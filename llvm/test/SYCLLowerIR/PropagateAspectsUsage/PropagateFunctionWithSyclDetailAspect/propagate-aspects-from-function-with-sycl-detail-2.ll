; RUN opt -passes=sycl-propagate-aspects-usage %s -S | FileCheck %s

; Check: that baz() takes a mix of "!sycl_used_aspects" of bar() & foo()

;    baz()
;   /     \
;  v       v
; bar()   foo()

source_filename = "main.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class.anon = type { i8 }

define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_() {
entry:
  %__SYCLKernel = alloca %class.anon, align 1
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(%class.anon addrspace(4)* noundef align 1 dereferenceable_or_null(1) %__SYCLKernel.ascast)
  ret void
}

define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(%class.anon addrspace(4)* noundef align 1 dereferenceable_or_null(1) %this) align 2 {
entry:
  call spir_func void @_Z3bazv()
  ret void
}

; CHECK: void @_Z3bazv() !sycl_used_aspects ![[#ASPECT1:]] {
define dso_local spir_func void @_Z3bazv() {
entry:
  call spir_func void @_Z3barv()
  call spir_func void @_Z3foov()
  ret void
}

; CHECK: void @_Z3barv() !sycl_used_aspects ![[#ASPECT2:]] {
define dso_local spir_func void @_Z3barv() !sycl_used_aspects !39 {
entry:
  ret void
}

; CHECK: void @_Z3foov() !sycl_used_aspects ![[#ASPECT3:]] {
define dso_local spir_func void @_Z3foov() !sycl_used_aspects !40 {
entry:
  ret void
}

!sycl_aspects = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37}

; CHECK: ![[#ASPECT1]] = !{i32 2, i32 1}
; CHECK: ![[#ASPECT2]] = !{i32 2}
; CHECK: ![[#ASPECT3]] = !{i32 1}

!0 = !{!"host", i32 0}
!1 = !{!"cpu", i32 1}
!2 = !{!"gpu", i32 2}
!3 = !{!"accelerator", i32 3}
!4 = !{!"custom", i32 4}
!5 = !{!"fp16", i32 5}
!6 = !{!"fp64", i32 6}
!7 = !{!"image", i32 9}
!8 = !{!"online_compiler", i32 10}
!9 = !{!"online_linker", i32 11}
!10 = !{!"queue_profiling", i32 12}
!11 = !{!"usm_device_allocations", i32 13}
!12 = !{!"usm_host_allocations", i32 14}
!13 = !{!"usm_shared_allocations", i32 15}
!14 = !{!"usm_restricted_shared_allocations", i32 16}
!15 = !{!"usm_system_allocations", i32 17}
!16 = !{!"ext_intel_pci_address", i32 18}
!17 = !{!"ext_intel_gpu_eu_count", i32 19}
!18 = !{!"ext_intel_gpu_eu_simd_width", i32 20}
!19 = !{!"ext_intel_gpu_slices", i32 21}
!20 = !{!"ext_intel_gpu_subslices_per_slice", i32 22}
!21 = !{!"ext_intel_gpu_eu_count_per_subslice", i32 23}
!22 = !{!"ext_intel_max_mem_bandwidth", i32 24}
!23 = !{!"ext_intel_mem_channel", i32 25}
!24 = !{!"usm_atomic_host_allocations", i32 26}
!25 = !{!"usm_atomic_shared_allocations", i32 27}
!26 = !{!"atomic64", i32 28}
!27 = !{!"ext_intel_device_info_uuid", i32 29}
!28 = !{!"ext_oneapi_srgb", i32 30}
!29 = !{!"ext_oneapi_native_assert", i32 31}
!30 = !{!"host_debuggable", i32 32}
!31 = !{!"ext_intel_gpu_hw_threads_per_eu", i32 33}
!32 = !{!"ext_oneapi_cuda_async_barrier", i32 34}
!33 = !{!"ext_oneapi_bfloat16", i32 35}
!34 = !{!"ext_intel_free_memory", i32 36}
!35 = !{!"int64_base_atomics", i32 7}
!36 = !{!"int64_extended_atomics", i32 8}
!37 = !{!"usm_system_allocator", i32 17}
!38 = !{}
!39 = !{i32 2}
!40 = !{i32 1}
