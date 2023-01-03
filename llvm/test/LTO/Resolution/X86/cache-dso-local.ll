; Tests whether the cache is sensitive to the dso-local bit on referenced
; globals.
; RUN: rm -rf %t.cache
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers -module-hash -module-summary -o %t.bc %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto2 run -opaque-pointers -o %t.o %t.bc -cache-dir %t.cache \
; RUN:   -r %t.bc,foo,px \
; RUN:   -r %t.bc,bar,px
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto2 run -opaque-pointers -o %t.o %t.bc -cache-dir %t.cache \
; RUN:   -r %t.bc,foo,plx \
; RUN:   -r %t.bc,bar,px
; RUN: ls %t.cache | count 2

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define weak void @foo() {
  ret void
}

define weak ptr @bar() {
  ret ptr @foo
}
