; This test checks handling of sycl-grf-size in SYCL post link

; RUN: sycl-post-link -properties -split=source -symbols -split-esimd -lower-esimd -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes CHECK-ESIMD-LargeGRF-IR --implicit-check-not='__ESIMD_kernel()'
; RUN: FileCheck %s -input-file=%t_esimd_0.prop --check-prefixes CHECK-ESIMD-LargeGRF-PROP
; RUN: FileCheck %s -input-file=%t_esimd_0.sym --check-prefixes CHECK-ESIMD-LargeGRF-SYM
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-SYCL-LargeGRF-IR --implicit-check-not='__SYCL_kernel()'
; RUN: FileCheck %s -input-file=%t_1.prop --check-prefixes CHECK-SYCL-LargeGRF-PROP
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYCL-LargeGRF-SYM
; RUN: FileCheck %s -input-file=%t_3.ll --check-prefixes CHECK-SYCL-IR --implicit-check-not='__SYCL_kernel_large_grf()'
; RUN: FileCheck %s -input-file=%t_3.prop --check-prefixes CHECK-SYCL-PROP
; RUN: FileCheck %s -input-file=%t_3.sym --check-prefixes CHECK-SYCL-SYM
; RUN: FileCheck %s -input-file=%t_esimd_2.ll --check-prefixes CHECK-ESIMD-IR --implicit-check-not='__ESIMD_large_grf_kernel()'
; RUN: FileCheck %s -input-file=%t_esimd_2.prop --check-prefixes CHECK-ESIMD-PROP

; CHECK: [Code|Properties|Symbols]
; CHECK: {{.*}}_esimd_0.ll|{{.*}}_esimd_0.prop|{{.*}}_esimd_0.sym
; CHECK: {{.*}}_1.ll|{{.*}}_1.prop|{{.*}}_1.sym
; CHECK: {{.*}}_esimd_2.ll|{{.*}}_esimd_2.prop|{{.*}}_esimd_2.sym

; CHECK-ESIMD-LargeGRF-PROP: isEsimdImage=1|1
; CHECK-ESIMD-LargeGRF-PROP: sycl-grf-size=1|256

; CHECK-SYCL-LargeGRF-PROP: sycl-grf-size=1|256

; CHECK-SYCL-LargeGRF-IR: define {{.*}} spir_kernel void @__SYCL_kernel_large_grf() #[[SYCLAttr:]]
; CHECK-SYCL-LargeGRF-IR: attributes #[[SYCLAttr]]

; CHECK-SYCL-PROP-NOT: sycl-grf-size

; CHECK-SYCL-SYM: __SYCL_kernel
; CHECK-SYCL-SYM-EMPTY:

; CHECK-SYCL-IR: __SYCL_kernel() #[[SYCLAttr:]]
; CHECK-SYCL-IR: attributes #[[SYCLAttr]]

; CHECK-SYCL-LargeGRF-SYM: __SYCL_kernel_large_grf
; CHECK-SYCL-LargeGRF-SYM-EMPTY:

; CHECK-ESIMD-SYM: __ESIMD_kernel
; CHECK-ESIMD-SYM-EMPTY:

; CHECK-ESIMD-IR: __ESIMD_kernel() #[[ESIMDAttr:]]
; CHECK-ESIMD-IR: attributes #[[ESIMDAttr]]

; CHECK-ESIMD-PROP-NOT: sycl-grf-size

; CHECK-ESIMD-LargeGRF-SYM: __ESIMD_large_grf_kernel
; CHECK-ESIMD-LargeGRF-SYM-EMPTY:

; CHECK-ESIMD-LargeGRF-IR: @__ESIMD_large_grf_kernel() #[[ESIMDLargeAttr:]]
; CHECK-ESIMD-LargeGRF-IR: attributes #[[ESIMDLargeAttr]]

; ModuleID = 'large_grf.bc'
source_filename = "grf"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define weak_odr dso_local spir_kernel void @__SYCL_kernel() #0 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @__SYCL_kernel_large_grf() #1 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @__ESIMD_kernel() #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @__ESIMD_large_grf_kernel() #1 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
entry:
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-module-id"="a.cpp" "sycl-grf-size"="256" }

!0 = !{}
!1 = !{i32 1}