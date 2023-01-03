; RUN: llvm-as %S/Inputs/linkage.a.ll -o %t.1.bc
; RUN: llvm-as %S/Inputs/linkage.b.ll -o %t.2.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %t.1.bc  %t.2.bc
