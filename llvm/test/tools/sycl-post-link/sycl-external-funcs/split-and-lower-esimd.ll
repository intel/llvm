; RUN: sycl-post-link -properties -split=auto -split-esimd -lower-esimd -O0 -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll

; This test checks that unreferenced functions with sycl-module-id
; attribute are not dropped from the module and ESIMD lowering
; happens for them as well.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

declare dso_local spir_func void @_Z15__esimd_barrierv()

define dso_local spir_func void @externalESIMDDeviceFunc() #0 !sycl_explicit_simd !0 {
entry:
  call spir_func void @_Z15__esimd_barrierv()
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

!0 = !{}

; CHECK: define dso_local spir_func void @externalESIMDDeviceFunc()
; CHECK: entry:
; CHECK:   call void @llvm.genx.barrier()
; CHECK:   ret void
; CHECK: }
