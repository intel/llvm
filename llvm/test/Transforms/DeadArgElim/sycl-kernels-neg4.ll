; RUN: echo 'static constexpr const bool param_omit_table[] = {' > %t-int_header.h
; RUN: echo '  // OMIT_TABLE_BEGIN' >> %t-int_header.h
; RUN: echo '    // WrongKernelName' >> %t-int_header.h
; RUN: echo '    false, false,' >> %t-int_header.h
; RUN: echo '  // OMIT_TABLE_END' >> %t-int_header.h
; RUN: echo '};' >> %t-int_header.h
; RUN: not --crash opt < %s -deadargelim-sycl -S -integr-header-file %t-int_header.h

; Wrong kernel name in the integration header.

target triple = "spir64-unknown-unknown-sycldevice"

define weak_odr spir_kernel void @NegativeSpirKernel(float %arg1, float %arg2) {
  call void @foo(float %arg1)
  ret void
}

declare void @foo(float %arg)
