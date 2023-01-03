; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s
; PR3876
@gdtr = external global [0 x i8]

define void @test() {
	call zeroext i1 @paging_map(i64 zext (i32 and (i32 ptrtoint (ptr @gdtr to i32), i32 -4096) to i64))
	ret void
}

declare zeroext i1 @paging_map(i64)

