; Original code:
; #include <sycl/sycl.hpp>

; int main() {
;   sycl::queue q;
;   q.submit([&](sycl::handler &h) {
;       h.parallel_for<class KernelA>(
;           sycl::range<1>(32),
;           [=](sycl::item<1> it) [[sycl::reqd_work_group_size(32)]] {});
;     });
;   q.submit([&](sycl::handler &h) {
;       h.parallel_for<class KernelB>(
;           sycl::range<1>(32),
;           [=](sycl::item<1> it) [[sycl::reqd_work_group_size(64)]] {});
;     });
;   q.submit([&](sycl::handler &h) {
;       h.parallel_for<class KernelC>(
;           sycl::range<1>(32),
;           [=](sycl::item<1> it) [[sycl::reqd_work_group_size(32)]] {});
;     });
;   return 0;
; }

; RUN: sycl-post-link -properties -split=auto < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.prop --check-prefix CHECK-PROP-AUTO-SPLIT-0
; RUN: FileCheck %s -input-file=%t_1.prop --check-prefix CHECK-PROP-AUTO-SPLIT-1

; TODO: Before intel/llvm#10620, the reqd_work_group_size attribute
; stores its values as uint32_t, but this needed to be expanded to
; uint64_t.  However, this change did not happen in ABI-breaking
; window, so we attach the required work-group size as the
; reqd_work_group_size_uint64_t attribute. At the next ABI-breaking
; window, this can be changed back to reqd_work_group_size.
; CHECK-PROP-AUTO-SPLIT-0: [SYCL/device requirements]
; CHECK-PROP-AUTO-SPLIT-0-NEXT: aspects=2|AAAAAAAAAAA
; CHECK-PROP-AUTO-SPLIT-0-NEXT: reqd_work_group_size_uint64_t=2|ABAAAAAAAAAQAAAAAAAAAA

; CHECK-PROP-AUTO-SPLIT-1: [SYCL/device requirements]
; CHECK-PROP-AUTO-SPLIT-1-NEXT: aspects=2|AAAAAAAAAAA
; CHECK-PROP-AUTO-SPLIT-1-NEXT: reqd_work_group_size_uint64_t=2|ABAAAAAAAAAIAAAAAAAAAA

; ModuleID = '/tmp/source-5f7d0d.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

$_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ4mainENKUlRNS0_7handlerEE_clES4_E7KernelAEE = comdat any

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E7KernelA = comdat any

$_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ4mainENKUlRNS0_7handlerEE0_clES4_E7KernelBEE = comdat any

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E7KernelB = comdat any

$_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ4mainENKUlRNS0_7handlerEE1_clES4_E7KernelCEE = comdat any

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE1_clES2_E7KernelC = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ4mainENKUlRNS0_7handlerEE_clES4_E7KernelAEE() local_unnamed_addr #0 comdat !srcloc !46 !kernel_arg_buffer_location !47 !reqd_work_group_size !48 !sycl_fixed_targets !49 !sycl_kernel_omit_args !50 {
entry:
  %0 = load <3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, align 32, !noalias !51
  %1 = extractelement <3 x i64> %0, i64 0
  %cmp.i.i = icmp ult i64 %1, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E7KernelA() local_unnamed_addr #0 comdat !srcloc !60 !kernel_arg_buffer_location !49 !reqd_work_group_size !48 !sycl_fixed_targets !49 !sycl_kernel_omit_args !49 {
entry:
  ret void
}

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ4mainENKUlRNS0_7handlerEE0_clES4_E7KernelBEE() local_unnamed_addr #0 comdat !srcloc !46 !kernel_arg_buffer_location !47 !reqd_work_group_size !61 !sycl_fixed_targets !49 !sycl_kernel_omit_args !50 {
entry:
  %0 = load <3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, align 32, !noalias !62
  %1 = extractelement <3 x i64> %0, i64 0
  %cmp.i.i = icmp ult i64 %1, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  ret void
}

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE0_clES2_E7KernelB() local_unnamed_addr #0 comdat !srcloc !71 !kernel_arg_buffer_location !49 !reqd_work_group_size !61 !sycl_fixed_targets !49 !sycl_kernel_omit_args !49 {
entry:
  ret void
}

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTSN4sycl3_V16detail19__pf_kernel_wrapperIZZ4mainENKUlRNS0_7handlerEE1_clES4_E7KernelCEE() local_unnamed_addr #0 comdat !srcloc !46 !kernel_arg_buffer_location !47 !reqd_work_group_size !48 !sycl_fixed_targets !49 !sycl_kernel_omit_args !50 {
entry:
  %0 = load <3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, align 32, !noalias !72
  %1 = extractelement <3 x i64> %0, i64 0
  %cmp.i.i = icmp ult i64 %1, 2147483648
  tail call void @llvm.assume(i1 %cmp.i.i)
  ret void
}

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE1_clES2_E7KernelC() local_unnamed_addr #0 comdat !srcloc !81 !kernel_arg_buffer_location !49 !reqd_work_group_size !48 !sycl_fixed_targets !49 !sycl_kernel_omit_args !49 {
entry:
  ret void
}

attributes #0 = { norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="source.cpp" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!sycl_aspects = !{!2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42}
!llvm.ident = !{!43}
!llvm.module.flags = !{!44, !45}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"host", i32 0}
!3 = !{!"cpu", i32 1}
!4 = !{!"gpu", i32 2}
!5 = !{!"accelerator", i32 3}
!6 = !{!"custom", i32 4}
!7 = !{!"fp16", i32 5}
!8 = !{!"fp64", i32 6}
!9 = !{!"image", i32 9}
!10 = !{!"online_compiler", i32 10}
!11 = !{!"online_linker", i32 11}
!12 = !{!"queue_profiling", i32 12}
!13 = !{!"usm_device_allocations", i32 13}
!14 = !{!"usm_host_allocations", i32 14}
!15 = !{!"usm_shared_allocations", i32 15}
!16 = !{!"usm_restricted_shared_allocations", i32 16}
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
!40 = !{!"int64_base_atomics", i32 7}
!41 = !{!"int64_extended_atomics", i32 8}
!42 = !{!"usm_system_allocator", i32 17}
!43 = !{!"clang version 16.0.0"}
!44 = !{i32 1, !"wchar_size", i32 4}
!45 = !{i32 7, !"frame-pointer", i32 2}
!46 = !{i32 8347054}
!47 = !{i32 -1, i32 -1}
!48 = !{i32 32}
!49 = !{}
!50 = !{i1 true, i1 true}
!51 = !{!52, !54, !56, !58}
!52 = distinct !{!52, !53, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!53 = distinct !{!53, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!54 = distinct !{!54, !55, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!55 = distinct !{!55, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!56 = distinct !{!56, !57, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!57 = distinct !{!57, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!58 = distinct !{!58, !59, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!59 = distinct !{!59, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!60 = !{i32 170}
!61 = !{i32 64}
!62 = !{!63, !65, !67, !69}
!63 = distinct !{!63, !64, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!64 = distinct !{!64, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!65 = distinct !{!65, !66, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!66 = distinct !{!66, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!67 = distinct !{!67, !68, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!68 = distinct !{!68, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!69 = distinct !{!69, !70, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!70 = distinct !{!70, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!71 = !{i32 352}
!72 = !{!73, !75, !77, !79}
!73 = distinct !{!73, !74, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv: %agg.result"}
!74 = distinct !{!74, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEE8initSizeEv"}
!75 = distinct !{!75, !76, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v: %agg.result"}
!76 = distinct !{!76, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN4sycl3_V12idILi1EEEEET0_v"}
!77 = distinct !{!77, !78, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv: %agg.result"}
!78 = distinct !{!78, !"_ZN4sycl3_V16detail7Builder7getItemILi1ELb1EEENSt9enable_ifIXT0_EKNS0_4itemIXT_EXT0_EEEE4typeEv"}
!79 = distinct !{!79, !80, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE: %agg.result"}
!80 = distinct !{!80, !"_ZN4sycl3_V16detail7Builder10getElementILi1ELb1EEEDTcl7getItemIXT_EXT0_EEEEPNS0_4itemIXT_EXT0_EEE"}
!81 = !{i32 534}
