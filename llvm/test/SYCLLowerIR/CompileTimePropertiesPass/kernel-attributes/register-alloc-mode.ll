; Check we create RegisterAllocMode metadata if there is a non-ESIMD kernel with that property
; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s --check-prefix CHECK-IR

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @sycl_regallocmode() #1 {
; CHECK-IR-NOT: !RegisterAllocMode
; CHECK-IR: sycl_regallocmode() #[[#Attr1:]]{{.*}}!RegisterAllocMode ![[#MDVal:]] {
; CHECK-IR-NOT: !RegisterAllocMode
; CHECK-IR: ![[#MDVal]] = !{i32 2}
entry:
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @sycl_noregallocmode() #0 {
entry:
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @esimd_regallocmode() #1 !sycl_explicit_simd !1 {
entry:
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @esimd_noregallocmode() #0 {
entry:
  ret void
}

attributes #0 = { convergent norecurse }
attributes #1 = { convergent norecurse "sycl-register-alloc-mode"="2" }

!1 = !{}
