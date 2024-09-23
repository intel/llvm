; This test checks handling of RegisterAllocMode in SYCL post link

; RUN: sycl-post-link -properties -split=source -symbols -split-esimd -lower-esimd -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t.table

; CHECK: [Code|Properties|Symbols]
; CHECK-NEXT: {{.*}}_esimd_0.ll|{{.*}}_esimd_0.prop|{{.*}}_esimd_0.sym
; CHECK-NEXT: {{.*}}_1.ll|{{.*}}_1.prop|{{.*}}_1.sym
; CHECK-NEXT: {{.*}}_esimd_2.ll|{{.*}}_esimd_2.prop|{{.*}}_esimd_2.sym
; CHECK-NEXT: {{.*}}_3.ll|{{.*}}_3.prop|{{.*}}_3.sym

; RUN: FileCheck %s -input-file=%t_esimd_0.ll   --check-prefixes CHECK-ESIMD-LargeGRF-IR
; RUN: FileCheck %s -input-file=%t_esimd_0.prop --check-prefixes CHECK-ESIMD-LargeGRF-PROP
; RUN: FileCheck %s -input-file=%t_esimd_0.sym  --check-prefixes CHECK-ESIMD-LargeGRF-SYM

; CHECK-ESIMD-LargeGRF-SYM: __ESIMD_large_grf_kernel
; CHECK-ESIMD-LargeGRF-SYM-EMPTY:

; CHECK-ESIMD-LargeGRF-PROP: isEsimdImage=1|1
; CHECK-ESIMD-LargeGRF-PROP: sycl-register-alloc-mode=1|2

; RUN: FileCheck %s -input-file=%t_1.ll   --check-prefixes CHECK-SYCL-LargeGRF-IR
; RUN: FileCheck %s -input-file=%t_1.prop --check-prefixes CHECK-SYCL-LargeGRF-PROP
; RUN: FileCheck %s -input-file=%t_1.sym  --check-prefixes CHECK-SYCL-LargeGRF-SYM

; CHECK-SYCL-LargeGRF-SYM: __SYCL_kernel_large_grf
; CHECK-SYCL-LargeGRF-SYM-EMPTY:

; CHECK-SYCL-LargeGRF-PROP: sycl-register-alloc-mode=1|2

; RUN: FileCheck %s -input-file=%t_esimd_2.prop --check-prefixes CHECK-ESIMD-PROP 
; RUN: FileCheck %s -input-file=%t_esimd_2.sym  --check-prefixes CHECK-ESIMD-SYM 

; CHECK-ESIMD-SYM: __ESIMD_kernel
; CHECK-ESIMD-SYM-EMPTY:

; CHECK-ESIMD-PROP-NOT: sycl-register-alloc-mode

; RUN: FileCheck %s -input-file=%t_3.prop --check-prefixes CHECK-SYCL-PROP
; RUN: FileCheck %s -input-file=%t_3.sym  --check-prefixes CHECK-SYCL-SYM

; CHECK-SYCL-SYM: __SYCL_kernel
; CHECK-SYCL-SYM-EMPTY:

; CHECK-SYCL-PROP-NOT: sycl-register-alloc-mode



; ModuleID = 'large_grf.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define weak_odr dso_local spir_kernel void @__SYCL_kernel() #0 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @__SYCL_kernel_large_grf() #1 {
; CHECK-SYCL-LargeGRF-IR: define {{.*}} spir_kernel void @__SYCL_kernel_large_grf() #[[#Attr:]]
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @__ESIMD_kernel() #0 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {

entry:
  ret void
}

define weak_odr dso_local spir_kernel void @__ESIMD_large_grf_kernel() #1 !sycl_explicit_simd !0 !intel_reqd_sub_group_size !1 {
; CHECK-ESIMD-LargeGRF-IR: @__ESIMD_large_grf_kernel()
entry:
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-module-id"="a.cpp" "sycl-register-alloc-mode"="2" }

!0 = !{}
!1 = !{i32 1}
