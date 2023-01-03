; RUN: llvm-as -o %t %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto2 run -opaque-pointers %t -r %t,foo, -r %t,baz,p -o %t2 -save-temps
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers -o - %t2.0.0.preopt.bc | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--fuchsia"

; CHECK: declare void @foo()
@foo = weak alias void(), ptr @bar

define internal void @bar() {
  ret void
}

define ptr @baz() {
  ret ptr @foo
}
