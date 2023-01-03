; RUN: llvm-as -o %t.bc %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto -opaque-pointers -save-merged-module -o %t2 %t.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers -o - %t2.merged.bc | FileCheck %s

; CHECK-NOT: global i32

target triple = "x86_64-unknown-linux-gnu"

@0 = private global i32 42
@foo = constant ptr @0
