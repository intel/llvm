; RUN: llvm-as < %s > /dev/null
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

define void @t() {
entry:
     ret void

loop:           ; preds = %loop
     %tmp.4.i9 = getelementptr i32, ptr null, i32 %tmp.5.i10             ; <ptr> [#uses=1]
     %tmp.5.i10 = load i32, ptr %tmp.4.i9                ; <i32> [#uses=1]
     br label %loop
}
