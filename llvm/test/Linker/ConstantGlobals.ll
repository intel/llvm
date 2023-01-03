; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %s %S/Inputs/ConstantGlobals.ll -S | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %S/Inputs/ConstantGlobals.ll %s -S | FileCheck %s

; CHECK-DAG: @X = constant [1 x i32] [i32 8]
@X = external global [1 x i32]

; CHECK-DAG: @Y = external global [1 x i32]
@Y = external global [1 x i32]

define ptr @use-Y() {
  ret ptr @Y
}
