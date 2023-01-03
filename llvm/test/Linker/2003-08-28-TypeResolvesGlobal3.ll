; RUN: llvm-as < %s > %t.out1.bc
; RUN: echo "%M = type i32" | llvm-as > %t.out2.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %t.out2.bc %t.out1.bc

%M = type opaque

; GLobal using the resolved function prototype
@0 = global ptr @foo		; <ptr>:0 [#uses=0]

define void @foo.upgrd.1(ptr %V) {
	ret void
}

declare void @foo(ptr)

