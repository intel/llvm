; This test checks handling of unreferenced functions with sycl-module-id
; attribute with splitting in per-kernel mode.

; RUN: sycl-post-link -properties -split=kernel -symbols -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-IR0
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM0
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-IR1
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYM1
; RUN: FileCheck %s -input-file=%t_2.ll --check-prefixes CHECK-IR2
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefixes CHECK-SYM2

; RUN: sycl-post-link -properties -split=kernel -emit-only-kernels-as-entry-points -symbols -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_1.ll --check-prefixes CHECK-IR1
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYM1
; RUN: FileCheck %s -input-file=%t_0.ll --check-prefixes CHECK-IR0
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM0
; RUN: FileCheck %s -input-file=%t.table --check-prefixes CHECK-TABLE

; CHECK-TABLE: [Code|Properties|Symbols]
; CHECK-TABLE-NEXT: {{.*}}_0.ll|{{.*}}_0.prop|{{.*}}_0.sym
; CHECK-TABLE-NEXT: {{.*}}_1.ll|{{.*}}_1.prop|{{.*}}_1.sym
; CHECK-TABLE-EMPTY:

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux"

define dso_local spir_func void @externalDeviceFunc() #0 {
  ret void
}

define dso_local spir_kernel void @kernel1() #0 {
  ret void
}

define dso_local spir_kernel void @kernel2() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

; CHECK-IR2: define dso_local spir_func void @externalDeviceFunc()
; CHECK-IR1: define dso_local spir_kernel void @kernel1()
; CHECK-IR0: define dso_local spir_kernel void @kernel2()

; CHECK-SYM2: externalDeviceFunc
; CHECK-SYM2-EMPTY:
; CHECK-SYM1: kernel1
; CHECK-SYM1-EMPTY:
; CHECK-SYM0: kernel2
; CHECK-SYM0-EMPTY:
