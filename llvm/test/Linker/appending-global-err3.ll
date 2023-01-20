; RUN: not llvm-link -opaque-pointers %s %p/Inputs/appending-global.ll -S -o - 2>&1 | FileCheck %s
; RUN: not llvm-link -opaque-pointers %p/Inputs/appending-global.ll %s -S -o - 2>&1 | FileCheck %s

; Negative test to check that global variables with appending linkage and
; different element types cannot be linked.

; CHECK: Appending variables with different element types

@var = appending global [1 x i32] undef
