; This test checks handling of the
;   __sycl_set_kernel_properties(...);
; intrinsic by LowerKernelProps pass - it should:
; - determine kernels calling this intrinsic (walk up the call graph)
; - remove the intrinsic call 
; - mark the kernel with corresponding attribute (only "large-grf" for now)

; RUN: opt -passes=lower-kernel-props -S %s -o - | FileCheck %s

; ModuleID = 'large_grf.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define dso_local spir_func void @_Z17large_grf_markerv() {
; CHECK: define dso_local spir_func void @_Z17large_grf_markerv()
; -- '0' constant argument means "large GRF" property:
  call spir_func void @_Z28__sycl_set_kernel_propertiesi(i32 noundef 0)
; -- Check that LowerKernelProps removed the marker call above:
; CHECK-NOT: {{.*}} @_Z28__sycl_set_kernel_propertiesi
  ret void
; CHECK-NEXT: ret void
}

declare dso_local spir_func void @_Z28__sycl_set_kernel_propertiesi(i32 noundef)

; -- This kernel calls the marker function indirectly
define weak_odr dso_local spir_kernel void @__large_grf_kernel1() !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
; CHECK: {{.*}} spir_kernel void @__large_grf_kernel1() #0
  call spir_func void @_Z17large_grf_markerv()
  ret void
}

; -- This kernel calls the marker function directly
define weak_odr dso_local spir_kernel void @__large_grf_kernel2() #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
; CHECK: {{.*}} spir_kernel void @__large_grf_kernel2() #0
  call spir_func void @_Z28__sycl_set_kernel_propertiesi(i32 noundef 0)
  ret void
}

attributes #0 = { "large-grf" }

!0 = !{}
!1 = !{i32 1}
