; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers -module-summary %s -o %t.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: opt -opaque-pointers -module-summary %p/Inputs/funcimport_appending_global.ll -o %t2.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-lto -opaque-pointers -thinlto -o %t3 %t.bc %t2.bc

; Do the import now
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %t.bc -summary-index=%t3.thinlto.bc -import=foo:%t2.bc -S | FileCheck %s

; Ensure that global constructor (appending linkage) is not imported
; CHECK-NOT: @llvm.global_ctors = {{.*}}@foo

declare void @f()
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @f, ptr null}]

define i32 @main() {
entry:
  call void @foo()
  ret i32 0
}

declare void @foo()
