; This test confirms an error with sycl-register-alloc-mode and sycl-grf-size on the same kernel.

; RUN: not sycl-post-link -split=source -symbols -split-esimd -lower-esimd -S < %s 2>&1 | FileCheck %s

; CHECK: Unsupported use of both register_alloc_mode and grf_size

source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define weak_odr dso_local spir_kernel void @__SYCL_kernel() #0 {
entry:
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" "sycl-grf-size"="256" "sycl-register-alloc-mode"="0"}
