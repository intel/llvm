; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llc -opaque-pointers -filetype=obj -o %t %p/Inputs/foo-return-i32-0.ll
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: lli -opaque-pointers -jit-kind=orc-lazy -extra-object %t %s
;
; Check that we can load an object file and call a function in it.

declare i32 @foo()

define i32 @main(i32 %argc, ptr %argv) {
entry:
  %0 = call i32 @foo()
  ret i32 %0
}

