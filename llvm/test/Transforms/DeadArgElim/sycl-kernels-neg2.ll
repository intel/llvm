; RUN: touch %t-int_header.h
; RUN: not --crash opt < %s -deadargelim-sycl -S -integr-header-file %t-bad_file.h

; Path to the integration header is wrong.

target triple = "spir64-unknown-unknown-sycldevice"

define weak_odr spir_kernel void @NegativeSpirKernel(float %arg1, float %arg2) {
  call void @foo(float %arg1)
  ret void
}

declare void @foo(float %arg)
