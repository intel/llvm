; RUN: sycl-post-link -split=kernel -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-IR1
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-IR2
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM1
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYM2

; This test checks that unreferenced functions with sycl-module-id
; attribute are not dropped from the module after splitting
; in per-kernel mode.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

define dso_local spir_func void @externalDeviceFunc() #0 {
  ret void
}

define dso_local spir_kernel void @kernel1() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

; CHECK-IR1: define dso_local spir_func void @externalDeviceFunc()
; CHECK-IR2: define dso_local spir_kernel void @kernel1()

; CHECK-SYM1: externalDeviceFunc
; CHECK-SYM2: kernel1
