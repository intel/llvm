; Test that opaque types are preserved correctly
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | llvm-as | llvm-dis -opaque-pointers
;
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

%Ty = type opaque

define ptr @func() {
	ret ptr null
}
 
