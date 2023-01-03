; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

; CHECK: @c = local_unnamed_addr constant i32 0
@c = local_unnamed_addr constant i32 0

; CHECK: @a = local_unnamed_addr alias i32, ptr @c
@a = local_unnamed_addr alias i32, ptr @c

; CHECK: define void @f() local_unnamed_addr {
define void @f() local_unnamed_addr {
  ret void
}
