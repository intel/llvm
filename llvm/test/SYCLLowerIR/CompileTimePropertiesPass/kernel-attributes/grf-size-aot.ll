; A GRF size of 512 has no RegisterAllocMode representation. In AOT (spir64_gen)
; it is lowered through MaximumRegisters (SPV_INTEL_maximum_registers) instead.
; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s

target triple = "spir64_gen-unknown-unknown"

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @sycl_grf_size_512() #0 {
; CHECK: sycl_grf_size_512() #[[#]]{{.*}}!MaximumRegisters ![[#MRVal:]] {
entry:
  ret void
}

; CHECK: ![[#MRVal]] = !{i32 512}

attributes #0 = { convergent norecurse "sycl-grf-size"="512" }
