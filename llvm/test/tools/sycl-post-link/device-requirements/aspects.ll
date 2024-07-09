; Original code:
; #include <sycl/sycl.hpp>

; [[__sycl_detail__::__uses_aspects__(sycl::aspect::fp64, sycl::aspect::cpu)]] void foo() {}

; [[__sycl_detail__::__uses_aspects__(sycl::aspect::queue_profiling, sycl::aspect::host, sycl::aspect::image)]] void bar() {}

; int main() {
;   sycl::queue q;
;   q.submit([&](sycl::handler &cgh) {
;     cgh.single_task([=]() { foo(); });
;     cgh.single_task([=]() { bar(); });
;   });
; }

; RUN: sycl-post-link -properties -split=auto < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefix CHECK-PROP-AUTO-SPLIT

; RUN: sycl-post-link -properties -split=kernel < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop --check-prefix CHECK-PROP-KERNEL-SPLIT-1
; RUN: FileCheck %s -input-file=%t.files_1.prop --check-prefix CHECK-PROP-KERNEL-SPLIT-0

; CHECK-PROP-AUTO-SPLIT: [SYCL/device requirements]
; CHECK-PROP-AUTO-SPLIT-NEXT: aspects=2|gCAAAAAAAAAAAAAABAAAAYAAAAQCAAAAMAAAAA

; CHECK-PROP-KERNEL-SPLIT-0: [SYCL/device requirements]
; CHECK-PROP-KERNEL-SPLIT-0-NEXT: aspects=2|gBAAAAAAAAAAAAAAJAAAAwAAAAA

; CHECK-PROP-KERNEL-SPLIT-1: [SYCL/device requirements]
; CHECK-PROP-KERNEL-SPLIT-1-NEXT: aspects=2|ABAAAAAAAAQAAAAAGAAAAA

source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class.anon = type { i8 }

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_ = comdat any

$_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE0_ = comdat any

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_() #0 comdat !kernel_arg_buffer_location !43 {
entry:
  %__SYCLKernel = alloca %class.anon, align 1
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(%class.anon addrspace(4)* noundef align 1 dereferenceable_or_null(1) %__SYCLKernel.ascast) #3
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv(%class.anon addrspace(4)* noundef align 1 dereferenceable_or_null(1) %this) #1 align 2 {
entry:
  %this.addr = alloca %class.anon addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %class.anon addrspace(4)** %this.addr to %class.anon addrspace(4)* addrspace(4)*
  store %class.anon addrspace(4)* %this, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %class.anon addrspace(4)*, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  call spir_func void @_Z3foov() #3
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @_Z3foov() #2 !sycl_used_aspects !44 {
entry:
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE0_() #0 comdat !kernel_arg_buffer_location !43 {
entry:
  %__SYCLKernel = alloca %class.anon, align 1
  %__SYCLKernel.ascast = addrspacecast %class.anon* %__SYCLKernel to %class.anon addrspace(4)*
  call spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE0_clEv(%class.anon addrspace(4)* noundef align 1 dereferenceable_or_null(1) %__SYCLKernel.ascast) #3
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define internal spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE0_clEv(%class.anon addrspace(4)* noundef align 1 dereferenceable_or_null(1) %this) #1 align 2 {
entry:
  %this.addr = alloca %class.anon addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %class.anon addrspace(4)** %this.addr to %class.anon addrspace(4)* addrspace(4)*
  store %class.anon addrspace(4)* %this, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %class.anon addrspace(4)*, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  call spir_func void @_Z3barv() #3
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func void @_Z3barv() #2 !sycl_used_aspects !45 {
entry:
  ret void
}

attributes #0 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="main2.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent }

!opencl.spir.version = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!spirv.Source = !{!1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1, !1}
!sycl_aspects = !{!2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39}
!llvm.ident = !{!40, !40, !40, !40, !40, !40, !40, !40, !40, !40, !40, !40, !40, !40, !40, !40}
!llvm.module.flags = !{!41, !42}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"host", i32 0}
!3 = !{!"cpu", i32 1}
!4 = !{!"gpu", i32 2}
!5 = !{!"accelerator", i32 3}
!6 = !{!"custom", i32 4}
!7 = !{!"fp16", i32 5}
!8 = !{!"fp64", i32 6}
!9 = !{!"int64_base_atomics", i32 7}
!10 = !{!"int64_extended_atomics", i32 8}
!11 = !{!"image", i32 9}
!12 = !{!"online_compiler", i32 10}
!13 = !{!"online_linker", i32 11}
!14 = !{!"queue_profiling", i32 12}
!15 = !{!"usm_device_allocations", i32 13}
!16 = !{!"usm_host_allocations", i32 14}
!17 = !{!"usm_shared_allocations", i32 15}
!18 = !{!"usm_restricted_shared_allocations", i32 16}
!19 = !{!"usm_system_allocations", i32 17}
!20 = !{!"usm_system_allocator", i32 17}
!21 = !{!"ext_intel_pci_address", i32 18}
!22 = !{!"ext_intel_gpu_eu_count", i32 19}
!23 = !{!"ext_intel_gpu_eu_simd_width", i32 20}
!24 = !{!"ext_intel_gpu_slices", i32 21}
!25 = !{!"ext_intel_gpu_subslices_per_slice", i32 22}
!26 = !{!"ext_intel_gpu_eu_count_per_subslice", i32 23}
!27 = !{!"ext_intel_max_mem_bandwidth", i32 24}
!28 = !{!"ext_intel_mem_channel", i32 25}
!29 = !{!"usm_atomic_host_allocations", i32 26}
!30 = !{!"usm_atomic_shared_allocations", i32 27}
!31 = !{!"atomic64", i32 28}
!32 = !{!"ext_intel_device_info_uuid", i32 29}
!33 = !{!"ext_oneapi_srgb", i32 30}
!34 = !{!"ext_oneapi_native_assert", i32 31}
!35 = !{!"host_debuggable", i32 32}
!36 = !{!"ext_intel_gpu_hw_threads_per_eu", i32 33}
!37 = !{!"ext_oneapi_cuda_async_barrier", i32 34}
!38 = !{!"ext_oneapi_bfloat16", i32 35}
!39 = !{!"ext_intel_free_memory", i32 36}
!40 = !{!"clang version 16.0.0"}
!41 = !{i32 1, !"wchar_size", i32 4}
!42 = !{i32 7, !"frame-pointer", i32 2}
!43 = !{}
!44 = !{i32 6, i32 1}
!45 = !{i32 12, i32 0, i32 9}
