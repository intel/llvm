; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-as < %s | llvm-dis -opaque-pointers | FileCheck %s

@foo = dso_local ifunc i32 (i32), ptr @foo_ifunc
; CHECK: @foo = dso_local ifunc i32 (i32), ptr @foo_ifunc

define internal ptr @foo_ifunc() {
entry:
  ret ptr null
}
