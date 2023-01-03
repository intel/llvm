; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | llvm-as | llvm-dis -opaque-pointers | FileCheck %s

; Test that align(N) is accepted as an alternative syntax to align N

; CHECK: define void @param_align4(ptr align 4 %ptr) {
define void @param_align4(ptr align(4) %ptr) {
  ret void
}

; CHECK: define void @param_align128(ptr align 128 %0) {
define void @param_align128(ptr align(128)) {
  ret void
}
