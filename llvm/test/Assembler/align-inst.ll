; RUN: llvm-as %s -o /dev/null
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

@A = global i1 0, align 4294967296

define void @foo() {
  %p = alloca i1, align 4294967296
  load i1, ptr %p, align 4294967296
  store i1 false, ptr %p, align 4294967296
  ret void
}
