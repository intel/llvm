; RUN: sycl-post-link -ir-output-only -split=auto -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll

; This test checks that unreferenced functions with sycl-module-id
; attribute are not dropped from the module after splitting
; in global mode.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

define dso_local spir_func void @externalDeviceFunc() #0 {
  ret void
}

define dso_local spir_kernel void @kernel1() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

; CHECK-DAG: define dso_local spir_func void @externalDeviceFunc()
; CHECK-DAG: define dso_local spir_kernel void @kernel1()
