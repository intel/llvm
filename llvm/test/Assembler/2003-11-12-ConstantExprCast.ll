; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | not grep " bitcast ("
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

@.Base64_1 = external constant [4 x i8]         ; <ptr> [#uses=1]

define i8 @test(i8 %Y) {
        %X = bitcast i8 %Y to i8                ; <i8> [#uses=1]
        %tmp.13 = add i8 %X, sub (i8 0, i8 ptrtoint (ptr @.Base64_1 to i8))     ; <i8> [#uses=1]
        ret i8 %tmp.13
}

