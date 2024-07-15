; Test that when the -support-dynamic-linking option is used with source splitting,
; dependencies to a function that can be imported do not cause the function
; to be added to a device image.
; Also ensure that functions in the same source that can be imported do not get split into
; different images.

; RUN: sycl-post-link -properties -symbols -support-dynamic-linking -split=source -S < %s -o %t.table

target triple = "spir64-unknown-unknown"

; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM-0
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYM-1

; RUN: FileCheck %s -input-file=%t_0.ll  --check-prefixes CHECK-LL-0
; RUN: FileCheck %s -input-file=%t_1.ll  --check-prefixes CHECK-LL-1

;; device image for source b.cpp
; CHECK-SYM-0: sycl_external_B2
; CHECK-SYM-0: sycl_external_B1
; CHECK-SYM-0: kernel_B
; CHECK-LL-0-DAG: declare spir_func void @sycl_external_A2() #0
; CHECK-LL-0-DAG: define weak_odr spir_func void @sycl_external_B2() #1 {
; CHECK-LL-0-DAG: define weak_odr spir_func void @sycl_external_B1() #1 {
; CHECK-LL-0-DAG: define weak_odr spir_kernel void @kernel_B() #1 {
; CHECK-LL-0: attributes #0 = { "sycl-module-id"="a.cpp" }
; CHECK-LL-0: attributes #1 = { "sycl-module-id"="b.cpp" }

;; device image for source a.cpp
; CHECK-SYM-1: sycl_external_A2
; CHECK-SYM-1: sycl_external_A1
; CHECK-SYM-1: kernel_A
; CHECK-LL-1-DAG: declare spir_func void @sycl_external_B2() #1
; CHECK-LL-1-DAG: define weak_odr spir_func void @sycl_external_A2() #0 {
; CHECK-LL-1-DAG: define weak_odr spir_func void @sycl_external_A1() #0 {
; CHECK-LL-1-DAG: define weak_odr spir_kernel void @kernel_A() #0 {
; CHECK-LL-1: attributes #0 = { "sycl-module-id"="a.cpp" }
; CHECK-LL-1: attributes #1 = { "sycl-module-id"="b.cpp" }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; FILE a.cpp
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define weak_odr spir_func void @sycl_external_A2() #0 {
  ret void
}

define weak_odr spir_func void @sycl_external_A1() #0 {
  call void @sycl_external_A2()
  call void @sycl_external_B2()  
  ret void
}

define weak_odr spir_kernel void @kernel_A() #0 {
  call void @sycl_external_A1()
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; FILE b.cpp
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define weak_odr spir_func void @sycl_external_B2() #1 {
  ret void
}

define weak_odr spir_func void @sycl_external_B1() #1 {
  call void @sycl_external_A2()
  call void @sycl_external_B2()
  ret void
}

define weak_odr spir_kernel void @kernel_B() #1 {
  call void @sycl_external_B1()
  ret void
}

attributes #1 = { "sycl-module-id"="b.cpp" }
