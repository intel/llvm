; This fails linking when it is linked with an empty file as the first object file

; RUN: llvm-as > %t.LinkTest.bc < /dev/null
; RUN: llvm-as < %s > %t.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %t.LinkTest.bc %t.bc

@work = global i32 4		; <ptr> [#uses=1]
@test = global ptr getelementptr (i32, ptr @work, i64 1)		; <ptr> [#uses=0]

