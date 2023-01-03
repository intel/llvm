; RUN: llvm-as < %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

        %struct.S_102 = type { float, float }

declare %struct.S_102 @f_102() nounwind

@callthis = external global ptr            ; <ptr> [#uses=50]


define void @foo() {
        store ptr @f_102, ptr @callthis, align 8
        ret void
}
