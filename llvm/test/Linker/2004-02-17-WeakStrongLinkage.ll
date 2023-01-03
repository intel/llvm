; RUN: llvm-as < %s > %t.out2.bc
; RUN: echo "@me = global ptr null" | llvm-as > %t.out1.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %t.out1.bc %t.out2.bc -o /dev/null

@me = weak global ptr null		; <ptr> [#uses=0]


