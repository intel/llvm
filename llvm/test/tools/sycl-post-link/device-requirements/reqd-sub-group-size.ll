; Original code:
; Compile with: clang++ -fsycl -fsycl-device-only -fno-sycl-instrument-device-code -D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ -S reqd-sub-group-size.cpp
; #include <sycl/sycl.hpp>

; int main() {
;   sycl::queue q;
;   q.submit([&](sycl::handler &h) {
;       h.parallel_for<class KernelA>(
;           sycl::range<1>(32),
;           [=](sycl::item<1> it) [[sycl::reqd_sub_group_size(16)]] {});
;     });
;   q.submit([&](sycl::handler &h) {
;       h.parallel_for<class KernelB>(
;           sycl::range<1>(32),
;           [=](sycl::item<1> it) [[sycl::reqd_sub_group_size(32)]] {});
;     });
;   q.submit([&](sycl::handler &h) {
;       h.parallel_for<class KernelC>(
;           sycl::range<1>(32),
;           [=](sycl::item<1> it) [[sycl::reqd_sub_group_size(16)]] {});
;     });
;   return 0;
; }

; RUN: sycl-post-link -properties -split=auto %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop --check-prefix CHECK-PROP-AUTO-SPLIT-0
; RUN: FileCheck %s -input-file=%t_1.prop --check-prefix CHECK-PROP-AUTO-SPLIT-1

; CHECK-PROP-AUTO-SPLIT-0: [SYCL/device requirements]
; CHECK-PROP-AUTO-SPLIT-0: reqd_sub_group_size=1|32

; CHECK-PROP-AUTO-SPLIT-1: [SYCL/device requirements]
; CHECK-PROP-AUTO-SPLIT-1: reqd_sub_group_size=1|16

; ModuleID = 'foo.cpp'
source_filename = "foo.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E7KernelA = comdat any

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E7KernelB = comdat any

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE1_clES2_E7KernelC = comdat any

; Function Attrs: norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E7KernelA() local_unnamed_addr #0 comdat !srcloc !48 !kernel_arg_buffer_location !49 !intel_reqd_sub_group_size !50 !sycl_fixed_targets !49 !sycl_kernel_omit_args !49 {
entry:
  ret void
}

; Function Attrs: norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E7KernelB() local_unnamed_addr #0 comdat !srcloc !51 !kernel_arg_buffer_location !49 !intel_reqd_sub_group_size !52 !sycl_fixed_targets !49 !sycl_kernel_omit_args !49 {
entry:
  ret void
}

; Function Attrs: norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE1_clES2_E7KernelC() local_unnamed_addr #0 comdat !srcloc !53 !kernel_arg_buffer_location !49 !intel_reqd_sub_group_size !50 !sycl_fixed_targets !49 !sycl_kernel_omit_args !49 {
entry:
  ret void
}

attributes #0 = { norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="foo.cpp" "sycl-optlevel"="2" "uniform-work-group-size"="true" }

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
!47 = !{!"clang version 17.0.0 (https://github.com/jzc/llvm eed5b5576bef314433e8ae7313620dae399c9d22)"}
!48 = !{i32 170}
!49 = !{}
!50 = !{i32 16}
!51 = !{i32 351}
!52 = !{i32 32}
!53 = !{i32 532}
