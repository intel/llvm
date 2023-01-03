; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | FileCheck %s

; Check that the ellipsis round trips.

%struct.A = type { i32 }

declare ptr @f(ptr, ...)

define ptr @f_thunk(ptr %this, ...) {
  %rv = musttail call ptr (ptr, ...) @f(ptr %this, ...)
  ret ptr %rv
}
; CHECK-LABEL: define ptr @f_thunk(ptr %this, ...)
; CHECK: %rv = musttail call ptr (ptr, ...) @f(ptr %this, ...)
