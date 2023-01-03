; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers
@bar = external global <2 x i32>                ; <ptr> [#uses=1]

define void @main() {
        store <2 x i32> < i32 0, i32 1 >, ptr @bar
        ret void
}

