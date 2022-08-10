; RUN: opt -LowerESIMD -S < %s | FileCheck %s

; This test checks that LowerESIMD pass sets the 'alwaysinline'
; attribute for all non-kernel functions.
; If the function already has noinline attribute -- honor that.

define spir_kernel void @EsimdKernel1() {
; CHECK: @EsimdKernel1(
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    call void @bar()
  call void @foo()
  call void @bar()
  ret void
}

define spir_kernel void @EsimdKernel2() {
; CHECK: @EsimdKernel2(
; CHECK-NEXT:    call void @foobar()
; CHECK-NEXT:    call void @noinline_func()
  call void @foobar()
  call void @noinline_func()
  ret void
}

define spir_func void @foo() {
; CHECK: @foo() #[[ATTR_INL:[0-9]+]]
  ret void
}

define spir_func void @bar() {
; CHECK: @bar() #[[ATTR_INL]]
; CHECK-NEXT:    call void @foobar
; CHECK-NEXT:    call void @noinline_func()
  call void @foobar()
  call void @noinline_func()
  ret void
}

define spir_func void @foobar() {
; CHECK: @foobar() #[[ATTR_INL]]
  ret void
}

define spir_func void @noinline_func() #0 {
; CHECK: @noinline_func() #[[ATTR_NOINL:[0-9]+]] {
  ret void
}

attributes #0 = { noinline }
; CHECK-DAG: attributes #[[ATTR_INL]] = { alwaysinline }
; CHECK-DAG: attributes #[[ATTR_NOINL]] = { noinline }
