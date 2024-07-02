; Test that when -support-dynamic-linking is used
; non SYCL-EXTERNAL functions are internalized.
; Variables must not be internalized.

; RUN: sycl-post-link -symbols -support-dynamic-linking -split=kernel -S < %s -o %t.table
; RUN: FileCheck %s -input-file=%t_0.sym --check-prefixes CHECK-SYM-0
; RUN: FileCheck %s -input-file=%t_0.ll  --check-prefixes CHECK-LL-0

; CHECK-SYM-0: foo0

; Non SYCL-EXTERNAL Functions are internalized
; foo0 is a SYCL-EXTERNAL function
; CHECK-LL-0-DAG: define weak_odr spir_kernel void @foo0() #0 {
; Internalize does not change available_externally
; CHECK-LL-0-DAG: define available_externally spir_func void @internalA() {
; CHECK-LL-0-DAG: define internal spir_func void @internalB() {
; CHECK-LL-0-DAG: define internal spir_func void @internalC() {
; CHECK-LL-0-DAG: define internal spir_func void @internalD() {
; CHECK-LL-0-DAG: define internal spir_func void @internalE() {
; CHECK-LL-0-DAG: define internal spir_func void @internalF() {
; private is already internalized
; CHECK-LL-0-DAG: define private spir_func void @internalG() {
; CHECK-LL-0-DAG: define internal spir_func void @internalH() {
; CHECK-LL-0-DAG: define internal spir_func void @internalI() {
; CHECK-LL-0-DAG: attributes #0 = { "sycl-module-id"="a.cpp" }

; Ensure variables are unchanged
; CHECK-LL-0-DAG: @ae = available_externally addrspace(1) global i32 79, align 4
; CHECK-LL-0-DAG: @i1 = addrspace(1) global i32 1, align 4
; CHECK-LL-0-DAG: @i2 = internal addrspace(1) global i32 2, align 4
; CHECK-LL-0-DAG: @i3 = addrspace(1) global i32 3, align 4
; CHECK-LL-0-DAG: @i4 = common addrspace(1) global i32 0, align 4
; CHECK-LL-0-DAG: @i5 = internal addrspace(1) global i32 0, align 4
; CHECK-LL-0-DAG: @color_table = addrspace(2) constant [2 x i32] [i32 0, i32 1], align 4
; CHECK-LL-0-DAG: @noise_table = external addrspace(2) constant [256 x i32]
; CHECK-LL-0-DAG: @w = addrspace(1) constant i32 0, align 4
; CHECK-LL-0-DAG: @f.color_table = internal addrspace(2) constant [2 x i32] [i32 2, i32 3], align 4
; CHECK-LL-0-DAG: @e = external addrspace(1) global i32
; CHECK-LL-0-DAG: @f.t = internal addrspace(1) global i32 5, align 4
; CHECK-LL-0-DAG: @f.stint = internal addrspace(1) global i32 0, align 4
; CHECK-LL-0-DAG: @f.inside = internal addrspace(1) global i32 0, align 4
; CHECK-LL-0-DAG: @f.b = internal addrspace(2) constant float 1.000000e+00, align 4

target triple = "spir64-unknown-unknown"

@ae = available_externally addrspace(1) global i32 79, align 4
@i1 = addrspace(1) global i32 1, align 4
@i2 = internal addrspace(1) global i32 2, align 4
@i3 = addrspace(1) global i32 3, align 4
@i4 = common addrspace(1) global i32 0, align 4
@i5 = internal addrspace(1) global i32 0, align 4
@color_table = addrspace(2) constant [2 x i32] [i32 0, i32 1], align 4
@noise_table = external addrspace(2) constant [256 x i32]
@w = addrspace(1) constant i32 0, align 4
@f.color_table = internal addrspace(2) constant [2 x i32] [i32 2, i32 3], align 4
@e = external addrspace(1) global i32
@f.t = internal addrspace(1) global i32 5, align 4
@f.stint = internal addrspace(1) global i32 0, align 4
@f.inside = internal addrspace(1) global i32 0, align 4
@f.b = internal addrspace(2) constant float 1.000000e+00, align 4

define available_externally spir_func void @internalA() {
  ret void
}

define dso_local            spir_func void @internalB() {
  ret void
}

define external             spir_func void @internalC() {
  ret void
}

define internal             spir_func void @internalD() {
  ret void
}

define linkonce             spir_func void @internalE() {
  ret void
}

define linkonce_odr         spir_func void @internalF() {
  ret void
}

define private              spir_func void @internalG() {
  ret void
}

define weak                 spir_func void @internalH() {
  ret void
}

define weak_odr             spir_func void @internalI() {
  ret void
}

define weak_odr spir_kernel void @foo0() #0 {
  call void @internalA()
  call void @internalB()
  call void @internalC()
  call void @internalD()
  call void @internalE()
  call void @internalF()
  call void @internalG()
  call void @internalH()
  call void @internalI()

  %1  = load i32, ptr addrspace(1) @ae
  %2  = load i32, ptr addrspace(1) @i1
  %3  = load i32, ptr addrspace(1) @i2
  %4  = load i32, ptr addrspace(1) @i3
  %5  = load i32, ptr addrspace(1) @i4
  %6  = load i32, ptr addrspace(1) @i5
  %7  = load i32, ptr addrspace(2) @color_table
  %8  = load i32, ptr addrspace(2) @noise_table
  %9  = load i32, ptr addrspace(1) @w
  %10 = load i32, ptr addrspace(2) @f.color_table
  %11 = load i32, ptr addrspace(1) @e
  %12 = load i32, ptr addrspace(1) @f.t
  %13 = load i32, ptr addrspace(1) @f.stint
  %14 = load i32, ptr addrspace(1) @f.inside
  %15 = load i32, ptr addrspace(2) @f.b


  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }
