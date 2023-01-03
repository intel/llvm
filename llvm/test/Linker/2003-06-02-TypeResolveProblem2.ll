; RUN: echo "%%T = type i32" | llvm-as > %t.1.bc
; RUN: llvm-as < %s > %t.2.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %t.1.bc %t.2.bc

%T = type opaque
@X = constant { ptr } zeroinitializer		; <ptr> [#uses=0]

