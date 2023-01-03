; Test linking two functions with different prototypes and two globals 
; in different modules. This is for PR411
; RUN: llvm-as %S/Inputs/basiclink.a.ll -o %t.foo.bc
; RUN: llvm-as %S/Inputs/basiclink.b.ll -o %t.bar.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %t.foo.bc %t.bar.bc -o %t.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %t.bar.bc %t.foo.bc -o %t.bc
