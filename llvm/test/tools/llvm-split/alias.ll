; RUN: llvm-split -o %t %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-dis -opaque-pointers -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; CHECK0-DAG: @afoo = alias [2 x ptr], ptr @foo
; CHECK1-DAG: @afoo = external global [2 x ptr]
@afoo = alias [2 x ptr], ptr @foo

; CHECK0-DAG: declare void @abar()
; CHECK1-DAG: @abar = alias void (), ptr @bar
@abar = alias void (), ptr @bar

@foo = global [2 x ptr] [ptr @bar, ptr @abar]

define void @bar() {
  store [2 x ptr] zeroinitializer, ptr @foo
  store [2 x ptr] zeroinitializer, ptr @afoo
  ret void
}
