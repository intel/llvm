; Check we don't crash when linking a global variable with appending linkage
; if the types in their elements don't have a straightforward mapping, forcing
; us to use bitcasts.

; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers %s -o %t1.o
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers %p/Inputs/appending-var-2.ll -o %t2.o

; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto2 run -opaque-pointers -o %t3.o %t1.o %t2.o -r %t1.o,bar, -r %t2.o,bar,px

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"foo.1" = type { i8, i8 }
declare dso_local i32 @bar(ptr nocapture readnone %this) local_unnamed_addr

@llvm.used = appending global [1 x ptr] [ptr @bar], section "llvm.metadata"
