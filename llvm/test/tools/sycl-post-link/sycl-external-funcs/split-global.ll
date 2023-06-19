; This test checks handling of unreferenced functions with sycl-module-id
; attribute with splitting in global mode.

; RUN: sycl-post-link -ir-output-only -split=auto -S < %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll --check-prefix=CHECK-ALL

; RUN: sycl-post-link -ir-output-only -emit-only-kernels-as-entry-points -split=auto -S < %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll --check-prefix=CHECK-KERNEL-ONLY --implicit-check-not @externalDeviceFunc


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

define dso_local spir_func void @externalDeviceFunc() #0 {
  ret void
}

define dso_local spir_kernel void @kernel1() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

; CHECK-ALL-DAG: define dso_local spir_func void @externalDeviceFunc()
; CHECK-ALL-DAG: define dso_local spir_kernel void @kernel1()
;
; CHECK-KERNEL-ONLY: define dso_local spir_kernel void @kernel1()
