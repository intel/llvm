; RUN: sycl-post-link -split=source -symbols -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-IR1
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-IR2
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM1
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYM2

; This test checks that the definition of function externalDeviceFunc is
; present in both resulting modules when per-source split is requested.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

define dso_local spir_func void @externalDeviceFunc() #0 {
  ret void
}

define dso_local spir_kernel void @kernel1() #1 {
  call void @externalDeviceFunc()
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-module-id"="b.cpp" }

; CHECK-IR1: define dso_local spir_func void @externalDeviceFunc()
; CHECK-IR2: define dso_local spir_func void @externalDeviceFunc()
; CHECK-IR2: define dso_local spir_kernel void @kernel1()

; CHECK-SYM1: externalDeviceFunc
; CHECK-SYM2: kernel1
