; In this testcase, the bytecode reader or writer is not correctly handling the
; ConstExpr reference.  Disassembling this program assembled yields invalid
; assembly (because there are placeholders still around), which the assembler
; dies on.

; There are two things that need to be fixed here.  Obviously assembling and
; disassembling this would be good, but in addition to that, the bytecode
; reader should NEVER produce a program "successfully" with placeholders still
; around!
;
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | llvm-as
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

@.LC0 = internal global [4 x i8] c"foo\00"		; <ptr> [#uses=1]
@X = global ptr null		; <ptr> [#uses=0]

declare i32 @puts(ptr)

define void @main() {
bb1:
	%reg211 = call i32 @puts( ptr @.LC0 )		; <i32> [#uses=0]
	ret void
}
