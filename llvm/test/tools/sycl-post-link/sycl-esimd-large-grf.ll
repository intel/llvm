; This test checks handling of the
;   set_kernel_properties(kernel_properties::use_large_grf);
; by the post-link-tool:
; - ESIMD/SYCL splitting happens as usual
; - ESIMD module is further split into callgraphs for entry points requesting
;   "large GRF" and callgraphs for entry points which are not
; - Compiler adds 'isLargeGRF' property to the ESIMD device binary
;   images requesting "large GRF" 

; RUN: sycl-post-link -split=source -symbols -split-esimd -lower-esimd -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table
; RUN: FileCheck %s -input-file=%t_esimd_large_grf_0.ll --check-prefixes CHECK-ESIMD-LargeGRF-IR
; RUN: FileCheck %s -input-file=%t_esimd_large_grf_0.prop --check-prefixes CHECK-ESIMD-LargeGRF-PROP
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYCL-SYM
; RUN: FileCheck %s -input-file=%t_esimd_1.sym --check-prefixes CHECK-ESIMD-SYM
; RUN: FileCheck %s -input-file=%t_esimd_large_grf_0.sym --check-prefixes CHECK-ESIMD-LargeGRF-SYM

; CHECK: [Code|Properties|Symbols]
; CHECK: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK: {{.*}}_1.ll|{{.*}}_1.prop|{{.*}}_1.sym
; CHECK: {{.*}}_2.ll|{{.*}}_2.prop|{{.*}}_2.sym

; CHECK-ESIMD-LargeGRF-PROP: isEsimdImage=1|1
; CHECK-ESIMD-LargeGRF-PROP: isLargeGRF=1|1

; CHECK-SYCL-SYM: __SYCL_kernel
; CHECK-SYCL-SYM-EMPTY:

; CHECK-ESIMD-SYM: __ESIMD_kernel
; CHECK-ESIMD-SYM-EMPTY:

; CHECK-ESIMD-LargeGRF-SYM: __ESIMD_large_grf_kernel
; CHECK-ESIMD-LargeGRF-SYM-EMPTY:

; ModuleID = 'large_grf.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define weak_odr dso_local spir_kernel void @__SYCL_kernel() #0 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @__ESIMD_kernel() #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
entry:
  ret void
}

define dso_local spir_func void @_Z17large_grf_markerv() {
entry:
  call spir_func void @_Z28__sycl_set_kernel_propertiesi(i32 noundef 0)
; -- Check that ESIMD lowering removed the marker call above:
; CHECK-ESIMD-LargeGRF-IR-NOT: {{.*}} @_Z28__sycl_set_kernel_propertiesi
  ret void
}

declare dso_local spir_func void @_Z28__sycl_set_kernel_propertiesi(i32 noundef)

define weak_odr dso_local spir_kernel void @__ESIMD_large_grf_kernel() #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
; CHECK-ESIMD-LargeGRF-IR: @__ESIMD_large_grf_kernel() {{.*}} !RegisterAllocMode ![[MetadataArg:[0-9]+]]
; CHECK-ESIMD-LargeGRF-IR: ![[MetadataArg]] = !{i32 2}
entry:
  call spir_func void @_Z17large_grf_markerv()
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

!0 = !{}
!1 = !{i32 1}
