; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: lli -opaque-pointers -jit-kind=orc-lazy %s
;
; Basic correctness check: We can make a call inside lazily JIT'd code.
; Compared to minimal.ll, this demonstrates that we can call through a stub.

define i32 @foo() {
entry:
  ret i32 0
}

define i32 @main(i32 %argc, ptr nocapture readnone %argv) {
entry:
  %0 = call i32() @foo()
  ret i32 %0
}
