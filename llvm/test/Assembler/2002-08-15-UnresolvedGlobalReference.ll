; RUN: llvm-as %s -o /dev/null
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

@.LC0 = internal global [12 x i8] c"hello world\00"             ; <ptr> [#uses=1]

define ptr @test() {
        ret ptr @.LC0
}

