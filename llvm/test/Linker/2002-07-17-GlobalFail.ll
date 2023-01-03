; RUN: llvm-as < %s > %t.bc
; RUN: echo | llvm-as > %t.tmp.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %t.tmp.bc %t.bc

@X = constant i32 5		; <ptr> [#uses=2]
@Y = internal global [2 x ptr] [ ptr @X, ptr @X ]		; <ptr> [#uses=0]


