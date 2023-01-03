; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers < %s -passes=strip -S | llvm-as | llvm-dis -opaque-pointers
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

; Stripping the name from A should not break references to it.
%A = type opaque
@g1 = external global %A
@g2 = global ptr @g1
