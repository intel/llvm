; This test checks handling of the
;   set_kernel_properties(kernel_properties::use_double_grf);
; by the post-link-tool:
; - ESIMD/SYCL splitting happens as usual
; - ESIMD module is further split into callgraphs for entry points requesting
;   "double GRF" and callgraphs for entry points which are not
; - Compiler adds 'isDoubleGRF' property to the ESIMD device binary
;   images requesting "double GRF" 

; RUN: sycl-post-link -split=source -symbols -split-esimd -lower-esimd -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table
; RUN: FileCheck %s -input-file=%t_esimd_x2grf_0.ll --check-prefixes CHECK-ESIMD-2xGRF-IR
; RUN: FileCheck %s -input-file=%t_esimd_x2grf_0.prop --check-prefixes CHECK-ESIMD-2xGRF-PROP
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYCL-SYM
; RUN: FileCheck %s -input-file=%t_esimd_0.sym --check-prefixes CHECK-ESIMD-SYM
; RUN: FileCheck %s -input-file=%t_esimd_x2grf_0.sym --check-prefixes CHECK-ESIMD-2xGRF-SYM

; CHECK: [Code|Properties|Symbols]
; CHECK: {{.*}}esimd_x2grf_0.ll|{{.*}}esimd_x2grf_0.prop|{{.*}}esimd_x2grf_0.sym
; CHECK: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK: {{.*}}esimd_0.ll|{{.*}}esimd_0.prop|{{.*}}esimd_0.sym

; CHECK-ESIMD-2xGRF-PROP: isEsimdImage=1|1
; CHECK-ESIMD-2xGRF-PROP: isDoubleGRF=1|1

; CHECK-SYCL-SYM: __SYCL_kernel
; CHECK-SYCL-SYM-EMPTY:

; CHECK-ESIMD-SYM: __ESIMD_kernel
; CHECK-ESIMD-SYM-EMPTY:

; CHECK-ESIMD-2xGRF-SYM: __ESIMD_double_grf_kernel
; CHECK-ESIMD-2xGRF-SYM-EMPTY:

; ModuleID = 'double_grf.bc'
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

define dso_local spir_func void @_Z17double_grf_markerv() {
entry:
  call spir_func void @_Z28__sycl_set_kernel_propertiesi(i32 noundef 0)
; -- Check that ESIMD lowering removed the marker call above:
; CHECK-ESIMD-2xGRF-IR-NOT: {{.*}} @_Z28__sycl_set_kernel_propertiesi
  ret void
}

declare dso_local spir_func void @_Z28__sycl_set_kernel_propertiesi(i32 noundef)

define weak_odr dso_local spir_kernel void @__ESIMD_double_grf_kernel() #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
entry:
  call spir_func void @_Z17double_grf_markerv()
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

!0 = !{}
!1 = !{i32 1}
