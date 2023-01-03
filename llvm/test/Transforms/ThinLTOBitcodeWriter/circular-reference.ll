; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis -opaque-pointers | FileCheck --check-prefix=M0 %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis -opaque-pointers | FileCheck --check-prefix=M1 %s

; M0: @g = external constant
; M1: @g = constant
@g = constant ptr @g, !type !0

!0 = !{i32 0, !"typeid"}
