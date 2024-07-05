; RUN: sycl-post-link -properties -split-esimd -lower-esimd -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent norecurse mustprogress
define dso_local spir_kernel void @_ZTSZ6calleriE12kernel_esimd() #0 !sycl_explicit_simd !3 {
entry:
  tail call spir_func void @_Z21__esimd_nbarrier_inith(i8 zeroext 7)
  ret void
}

!3 = !{}

declare dso_local spir_func void @_Z21__esimd_nbarrier_inith(i8 zeroext)
; CHECK: attributes #0 = { {{.*}}"VCNamedBarrierCount"="7"{{.*}} }

attributes #0 = { "sycl-module-id"="a.cpp" }
