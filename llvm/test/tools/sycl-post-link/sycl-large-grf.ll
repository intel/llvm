; This test checks handling of the
;   set_kernel_properties(kernel_properties::use_large_grf);
; by the post-link-tool:
; - ESIMD/SYCL splitting happens as usual
; - ESIMD module is further split into callgraphs for entry points requesting
;   "large GRF" and callgraphs for entry points which are not
; - Compiler adds 'isLargeGRF' property to the device binary
;   images requesting "large GRF" 

; RUN: sycl-post-link -split=source -symbols -split-esimd -lower-esimd -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table
; RUN: FileCheck %s -input-file=%t_large_grf_1.ll --check-prefixes CHECK-LARGE-GRF-IR
; RUN: FileCheck %s -input-file=%t_large_grf_1.prop --check-prefixes CHECK-LARGE-GRF-PROP
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYCL-SYM
; RUN: FileCheck %s -input-file=%t_large_grf_1.sym --check-prefixes CHECK-LARGE-GRF-SYM

; CHECK: [Code|Properties|Symbols]
; CHECK: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK: {{.*}}_large_grf_1.ll|{{.*}}_large_grf_1.prop|{{.*}}_large_grf_1.sym

; CHECK-LARGE-GRF-PROP: isLargeGRF=1|1

; CHECK-SYCL-SYM: __SYCL_kernel
; CHECK-SYCL-SYM-EMPTY:

; CHECK-LARGE-GRF-SYM: __large_grf_kernel
; CHECK-LARGE-GRF-SYM-EMPTY:

; ModuleID = 'large_grf.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define weak_odr dso_local spir_kernel void @__SYCL_kernel() #0 {
entry:
  ret void
}

define dso_local spir_func void @_Z17large_grf_markerv() {
entry:
  call spir_func void @_Z28__sycl_set_kernel_propertiesi(i32 noundef 0)
; -- Check that LowerKernelProps lowering removed the marker call above:
; CHECK-LARGE-GRF-IR-NOT: {{.*}} @_Z28__sycl_set_kernel_propertiesi
  ret void
}

declare dso_local spir_func void @_Z28__sycl_set_kernel_propertiesi(i32 noundef)

define weak_odr dso_local spir_kernel void @__large_grf_kernel() #0 {
entry:
  call spir_func void @_Z17large_grf_markerv()
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

!0 = !{}
!1 = !{i32 1}
