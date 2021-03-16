; RUN: opt -LowerESIMD -S < %s | FileCheck %s

; This test checks that LowerESIMD pass sets the 'alwaysinline'
; attribute for all non-kernel functions.

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
  call void @foobar()
  ret void
}

define spir_func void @foo() {
; CHECK: @foo() #[[ATTR:[0-9]+]]
  ret void
}

define spir_func void @bar() {
; CHECK: @bar() #[[ATTR]]
; CHECK-NEXT:    call void @foobar
  call void @foobar()
  ret void
}

define spir_func void @foobar() {
; CHECK: @foobar() #[[ATTR]]
  ret void
}

; CHECK: attributes #[[ATTR]] = { alwaysinline }
