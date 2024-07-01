; Test that when the -support-dynamic-linking option is used,
; dependencies to a function that can be imported do not cause the function
; to be added to a device image.

; RUN: sycl-post-link -properties -symbols -support-dynamic-linking -split=kernel -S < %s -o %t.table


; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM-0
; RUN: FileCheck %s -input-file=%t_1.sym --check-prefixes CHECK-SYM-1
; RUN: FileCheck %s -input-file=%t_2.sym --check-prefixes CHECK-SYM-2
; RUN: FileCheck %s -input-file=%t_3.sym --check-prefixes CHECK-SYM-3

; RUN: FileCheck %s -input-file=%t_0.ll  --check-prefixes CHECK-LL-0
; RUN: FileCheck %s -input-file=%t_1.ll  --check-prefixes CHECK-LL-1
; RUN: FileCheck %s -input-file=%t_2.ll  --check-prefixes CHECK-LL-2
; RUN: FileCheck %s -input-file=%t_3.ll  --check-prefixes CHECK-LL-3

; CHECK-SYM-0: foo1
; CHECK-LL-0: declare spir_func void @childA() #0
; CHECK-LL-0: define weak_odr spir_kernel void @foo1() #0 {
; CHECK-LL-0: attributes #0 = { "sycl-module-id"="a.cpp" }

; CHECK-SYM-1: foo0
; CHECK-LL-1: declare spir_func void @childA() #0
; CHECK-LL-1: define weak_odr spir_kernel void @foo0() #0 {
; CHECK-LL-1: attributes #0 = { "sycl-module-id"="a.cpp" }

; Function internal does not have a sycl-module-id. Thus it is not a SYCL External function
; and is included in the device image.
; Function __private starts with "__" and thus is included in the device image.
; CHECK-SYM-2: childB
; CHECK-LL-2: define weak_odr spir_func void @internal() {
; CHECK-LL-2: define weak_odr spir_func void @__private() #0 {
; CHECK-LL-2: define weak_odr spir_func void @childB() #0 {
; CHECK-LL-2: attributes #0 = { "sycl-module-id"="a.cpp" }

; CHECK-SYM-3: childA
; CHECK-LL-3: declare spir_func void @childB() #0
; CHECK-LL-3: define weak_odr spir_func void @childA() #0 {
; CHECK-LL-3: attributes #0 = { "sycl-module-id"="a.cpp" }

target triple = "spir64-unknown-unknown"

define weak_odr spir_func void @internal() {
  ret void
}

define weak_odr spir_func void @__private() #0 {
  ret void
}

define weak_odr spir_func void @childB() #0 {
  call void @internal()
  call void @__private()
  ret void
}

define weak_odr spir_func void @childA() #0 {
  call void @childB()
  ret void
}

define weak_odr spir_kernel void @foo0() #0 {
  call void @childA()
  ret void
}

define weak_odr spir_kernel void @foo1() #0 {
  call void @childA()
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
attributes #1 = { "sycl-module-id"="external.cpp" }
