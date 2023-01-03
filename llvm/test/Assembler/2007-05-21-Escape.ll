; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | llvm-as | llvm-dis -opaque-pointers
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "x86_64-apple-darwin8"
	%struct.bar = type { i32 }
	%struct.foo = type { i32 }

define i32 @"Func64"(ptr %F) {
entry:
	ret i32 1
}

define i32 @Func64(ptr %B) {
entry:
	ret i32 0
}

define i32 @test() {
entry:
	%tmp = tail call i32 @"Func64"( ptr null )		; <i32> [#uses=0]
	%tmp1 = tail call i32 @Func64( ptr null )		; <i32> [#uses=0]
	ret i32 undef
}
