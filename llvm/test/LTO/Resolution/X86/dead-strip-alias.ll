; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers -module-summary -o %t %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers -module-summary -o %t2 %S/Inputs/dead-strip-alias.ll
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto2 run -opaque-pointers %t -r %t,main,px -r %t,alias,p -r %t,external, \
; RUN:               %t2 -r %t2,external,p \
; RUN: -save-temps -o %t3
; RUN: llvm-nm %t3.2 | FileCheck %s

; CHECK: D external

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@alias = alias ptr, ptr @internal

@internal = internal global ptr @external
@external = external global i8

define ptr @main() {
  ret ptr @alias
}
