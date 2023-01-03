; RUN: llvm-as %s -o /dev/null
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

@X = external global ptr
@X1 = external global ptr 
@X2 = external global ptr

%T = type {i32}
