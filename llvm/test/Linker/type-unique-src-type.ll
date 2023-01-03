; RUN: llvm-as %s -o %t.bc
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers -S %t.bc -o - | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers -S %s -o - | FileCheck %s

; Test that we don't try to map %C.0 and C and then try to map %C to a new type.
; This used to happen when lazy loading since we wouldn't then identify %C
; as a destination type until it was too late.

; CHECK: %C.0 = type { %B }
; CHECK-NEXT: %B = type { %A }
; CHECK-NEXT: %A = type { i8 }

; CHECK: @g1 = global %C.0 zeroinitializer
; CHECK:  getelementptr %C.0, ptr null, i64 0, i32 0, i32 0

%A   = type { i8 }
%B   = type { %A }
%C   = type { %B }
%C.0 = type { %B }
define void @f1() {
  getelementptr %C, ptr null, i64 0, i32 0, i32 0
  ret void
}
@g1 = global %C.0 zeroinitializer
