; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | grep "align 1024"
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

define i32 @test(ptr %arg) {
entry:
        %tmp2 = load i32, ptr %arg, align 1024      ; <i32> [#uses=1]
        ret i32 %tmp2
}
