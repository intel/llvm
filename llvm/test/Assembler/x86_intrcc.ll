; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

; Make sure no arguments is accepted
; CHECK: define x86_intrcc void @no_args() {
define x86_intrcc void @no_args() {
  ret void
}

; CHECK: define x86_intrcc void @byval_arg(ptr byval(i32) %0) {
define x86_intrcc void @byval_arg(ptr byval(i32)) {
  ret void
}
