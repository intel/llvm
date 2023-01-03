; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | not grep ptrtoint
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s
; All of these should be eliminable


define i32 @foo() {
	ret i32 and (i32 ptrtoint (ptr @foo to i32), i32 1)
}

define i32 @foo2() {
	ret i32 and (i32 1, i32 ptrtoint (ptr @foo2 to i32))
}

define i1 @foo3() {
	ret i1 icmp ne (ptr @foo3, ptr null)
}
