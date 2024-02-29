; This ensures we remove optnone from ESIMD functions unless they are SIMT or we didn't split ESIMD code out.
; RUN: sycl-post-link -split-esimd -lower-esimd -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll --check-prefixes CHECK,CHECK-ESIMD-SPLIT

; RUN: sycl-post-link -lower-esimd -S < %s -o %t1.table
; RUN: FileCheck %s -input-file=%t1_esimd_0.ll --check-prefixes CHECK,CHECK-NO-ESIMD-SPLIT
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define dso_local spir_func void @esimd() #0 !sycl_explicit_simd !0 {
; CHECK: void @esimd() #[[#ESIMDAttr:]]
  ret void
}

define dso_local spir_func void @esimd_simt() #1 !sycl_explicit_simd !0 {
; CHECK: void @esimd_simt() #[[#ESIMDSIMTAttr:]]
  ret void
}

define dso_local spir_func void @sycl() #0 {
; CHECK: spir_func void @sycl() #[[#ESIMDAttr]]
  ret void
}
; CHECK-ESIMD-SPLIT: attributes #[[#ESIMDAttr]] =
; CHECK-ESIMD-SPLIT-NOT: optnone
; CHECK-NO-ESIMD-SPLIT: attributes #[[#ESIMDAttr]] = {{{.*}}optnone{{.*}}}
; CHECK: attributes #[[#ESIMDSIMTAttr]] = {{{.*}}optnone
; CHECK-SAME: "CMGenxSIMT"
; CHECK-SAME: }
attributes #0 = { noinline optnone convergent }
attributes #1 = { noinline optnone convergent "CMGenxSIMT"}
!0 = !{}
