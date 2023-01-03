; RUN: llvm-as %s -o /dev/null
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

define ptr @t1(ptr %X) {
        %W = getelementptr { float, i32 }, ptr %X, i32 20, i32 1            ; <ptr> [#uses=0]
        %X.upgrd.1 = getelementptr { float, i32 }, ptr %X, i64 20, i32 1            ; <ptr> [#uses=0]
        %Y = getelementptr { float, i32 }, ptr %X, i64 20, i32 1            ; <ptr> [#uses=1]
        %Z = getelementptr { float, i32 }, ptr %X, i64 20, i32 1            ; <ptr> [#uses=0]
        ret ptr %Y
}

