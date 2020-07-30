; TODO: this should be crashing (not --crash) after enabling assert
; RUN: opt < %s -deadargelim-sycl -S

; Path to the integration header is not specified.

target triple = "spir64-unknown-unknown-sycldevice"

define weak_odr spir_kernel void @NegativeSyclKernel(float %arg1, float %arg2) {
  call void @foo(float %arg1)
  ret void
}

declare void @foo(float %arg)
