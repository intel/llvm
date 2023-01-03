; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %S/Inputs/PR8300.a.ll %S/Inputs/PR8300.b.ll -o %t.bc
