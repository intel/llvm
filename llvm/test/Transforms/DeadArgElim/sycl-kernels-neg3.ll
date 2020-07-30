; RUN: rm -rf %t && mkdir -p %t
; RUN: echo 'static constexpr const bool param_omit_table[] = {' >> %t/int_header.h
; RUN: echo '    // NegativeSyclKernel' >> %t/int_header.h
; RUN: echo '    false, false,' >> %t/int_header.h
; RUN: echo '};' >> %t/int_header.h
; RUN: not --crash opt < %s -deadargelim-sycl -S -integr-header-file %t/int_header.h

; No OMIT_TABLE markers in the integration header.

target triple = "spir64-unknown-unknown-sycldevice"

define weak_odr spir_kernel void @NegativeSyclKernel(float %arg1, float %arg2) {
  call void @foo(float %arg1)
  ret void
}

declare void @foo(float %arg)
