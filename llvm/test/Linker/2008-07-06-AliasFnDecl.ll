; PR2146
; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/2008-07-06-AliasFnDecl2.ll -o %t2.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %t1.bc %t2.bc -o %t3.bc

@b = alias void (), ptr @a

define void @a() nounwind  {
entry:
	br label %return

return:
	ret void
}
