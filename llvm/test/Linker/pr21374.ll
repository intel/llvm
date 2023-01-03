; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers -S -o - %p/pr21374.ll %p/Inputs/pr21374.ll | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers -S -o - %p/Inputs/pr21374.ll %p/pr21374.ll | FileCheck %s

; RUN: llvm-as -o %t1.bc %p/pr21374.ll
; RUN: llvm-as -o %t2.bc %p/Inputs/pr21374.ll

; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers -S -o - %t1.bc %t2.bc | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers -S -o - %t2.bc %t1.bc | FileCheck %s

; Test that we get the same result with or without lazy loading.

; CHECK: %foo = type { ptr }
; CHECK-DAG: getelementptr %foo, ptr null, i64 1
; CHECK-DAG: define void @g(%foo %x)

%foo = type { ptr }
define void @f() {
  getelementptr %foo, ptr null, i64 1
  ret void
}
