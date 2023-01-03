; This fails linking when it is linked with an empty file as the first object file

; RUN: llvm-as > %t1.bc < /dev/null
; RUN: llvm-as < %s  > %t2.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %t1.bc %t2.bc

@work = global ptr @zip		; <ptr> [#uses=0]

declare i32 @zip(i32, i32)

