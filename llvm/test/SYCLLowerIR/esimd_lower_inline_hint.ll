; RUN: opt -LowerESIMD -S < %s | FileCheck -check-prefixes=CHECK,CHECK-ATTR %s
; RUN: opt -LowerESIMD -lower-esimd-opt-level-O0 -S < %s | FileCheck -check-prefixes=CHECK,CHECK-NO-ATTR %s

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
; CHECK-ATTR: @foo() #[[ATTR:[0-9]+]] {
; CHECK-NO-ATTR @foo() {
  ret void
}

define spir_func void @bar() {
; CHECK-ATTR: @bar() #[[ATTR]] {
; CHECK-ATTR-NEXT:    call void @foobar
; CHECK-NO-ATTR: @bar() {
; CHECK-NO-ATTR-NEXT:    call void @foobar
  call void @foobar()
  ret void
}

define spir_func void @foobar() {
; CHECK-ATTR: @foobar() #[[ATTR]] {
; CHECK-NO-ATTR: @foobar() {
  ret void
}

; CHECK-ATTR: attributes #[[ATTR]] = { alwaysinline }
; CHECK-NO-ATTR-NOT: attributes {{.*}} alwaysinline
