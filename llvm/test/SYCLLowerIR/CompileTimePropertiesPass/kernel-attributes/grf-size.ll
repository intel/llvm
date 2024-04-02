; Check we create RegisterAllocMode metadata if there is a non-ESIMD kernel with that property
; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s --check-prefix CHECK-IR

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @sycl_grf_size() #1 {
; CHECK-IR-NOT: !RegisterAllocMode
; CHECK-IR: sycl_grf_size() #[[#Attr1:]]{{.*}}!RegisterAllocMode ![[#MDVal:]] {
; CHECK-IR-NOT: !RegisterAllocMode
; CHECK-IR: ![[#MDVal]] = !{i32 2}
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

attributes #0 = { convergent norecurse }
attributes #1 = { convergent norecurse "sycl-grf-size"="256" }

!1 = !{}