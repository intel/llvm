; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s -o - | llc -opaque-pointers -filetype=asm | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define internal ptr @foo_ifunc() {
entry:
  ret ptr null
}
; CHECK: .type foo_ifunc,@function
; CHECK-NEXT: foo_ifunc:

@foo = ifunc i32 (i32), ptr @foo_ifunc
; CHECK:      .type foo,@gnu_indirect_function
; CHECK-NEXT: .set foo, foo_ifunc
