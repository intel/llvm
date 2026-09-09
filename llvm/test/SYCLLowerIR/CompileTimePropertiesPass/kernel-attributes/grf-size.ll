; Check we create RegisterAllocMode metadata if there is a non-ESIMD kernel with that property
; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s --check-prefix CHECK-IR

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @sycl_grf_size() #1 {
; CHECK-IR-NOT: !RegisterAllocMode
; CHECK-IR: sycl_grf_size() #[[#Attr1:]]{{.*}}!RegisterAllocMode ![[#MDVal:]] {
; CHECK-IR-NOT: !RegisterAllocMode
entry:
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @sycl_no_grf_size() #0 {
entry:
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @esimd_grf_size() #1 !sycl_explicit_simd !1 {
entry:
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @esimd_no_grf_size() #0 {
entry:
  ret void
}

; A GRF size of 512 has no RegisterAllocMode representation. For JIT (no AOT
; triple) it is left to the runtime's -ze-opt-register-file-size=512 driver
; option, so no metadata is added here. See grf-size-aot.ll for the AOT case.
; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @sycl_grf_size_512() #2 {
; CHECK-IR-NOT: !MaximumRegisters
; CHECK-IR: sycl_grf_size_512() #[[#]] {
entry:
  ret void
}

; CHECK-IR-DAG: ![[#MDVal]] = !{i32 2}

attributes #0 = { convergent norecurse }
attributes #1 = { convergent norecurse "sycl-grf-size"="256" }
attributes #2 = { convergent norecurse "sycl-grf-size"="512" }

!1 = !{}
