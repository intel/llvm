; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers -S -preserve-ll-uselistorder < %s | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: verify-uselistorder -opaque-pointers %s

; CHECK: @g = external global i32
; CHECK: define void @func1() {
; CHECK-NOT: uselistorder
; CHECK: }
; CHECK: define void @func2() {
; CHECK-NOT: uselistorder
; CHECK: }
; CHECK: uselistorder ptr @g, { 3, 2, 1, 0 }

@g = external global i32

define void @func1() {
  load i32, ptr @g
  load i32, ptr @g
  ret void
}

define void @func2() {
  load i32, ptr @g
  load i32, ptr @g
  ret void
}

uselistorder ptr @g, { 3, 2, 1, 0 }
