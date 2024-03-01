; Check we don't assert for different GRF values and don't add the RegisterAllocMode metadata
; RUN: opt -passes=compile-time-properties %s -S | FileCheck %s --implicit-check-not=RegisterAllocMode

; CHECK: spir_kernel void @foo()
define weak_odr dso_local spir_kernel void @foo() #0 {
entry:
  ret void
}

attributes #0 = { convergent mustprogress norecurse nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-grf-size"="16384" "sycl-module-id"="main.cpp" "sycl-optlevel"="2" "uniform-work-group-size"="true" }
