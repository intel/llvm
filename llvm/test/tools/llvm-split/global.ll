; RUN: llvm-split -o %t %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0: @foo = global ptr @bar
; CHECK1: @foo = external global ptr
@foo = global ptr @bar

; CHECK0: @bar = external global ptr
; CHECK1: @bar = global ptr @foo
@bar = global ptr @foo
