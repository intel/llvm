; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llc -opaque-pointers -filetype=obj -o %t1.o %p/Inputs/obj-weak-non-materialization-1.ll
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llc -opaque-pointers -filetype=obj -o %t2.o %p/Inputs/obj-weak-non-materialization-2.ll
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: lli -opaque-pointers -jit-kind=orc-lazy -extra-object %t1.o -extra-object %t2.o %s
;
; Check that %t1.o's version of the weak symbol X is used, even though %t2.o is
; materialized first.

@X = external global i32

declare void @foo()

define i32 @main(i32 %argc, ptr %argv) {
entry:
  call void @foo()
  %0 = load i32, ptr @X
  ret i32 %0
}
