; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | grep byval
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

	%struct = type {  }

declare void @foo(...)

define void @bar() {
	call void (...) @foo(ptr byval(%struct) null )
	ret void
}
