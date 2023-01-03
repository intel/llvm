; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %s %p/Inputs/constructor-comdat.ll -S -o - 2>&1 | FileCheck %s
; Added -opaque-pointers.
; FIXME: Align with the community code when project is ready to enable opaque
; pointers by default
; RUN: llvm-link -opaque-pointers %p/Inputs/constructor-comdat.ll %s -S -o - 2>&1 | FileCheck --check-prefix=NOCOMDAT %s

$_ZN3fooIiEC5Ev = comdat any
; CHECK: $_ZN3fooIiEC5Ev = comdat any
; NOCOMDAT-NOT: comdat

@_ZN3fooIiEC1Ev = weak_odr alias void (), ptr @_ZN3fooIiEC2Ev
; CHECK: @_ZN3fooIiEC1Ev = weak_odr alias void (), ptr @_ZN3fooIiEC2Ev
; NOCOMDAT-DAG: define weak_odr void @_ZN3fooIiEC1Ev() {

; CHECK: define weak_odr void @_ZN3fooIiEC2Ev() comdat($_ZN3fooIiEC5Ev) {
; NOCOMDAT-DAG: define weak_odr void @_ZN3fooIiEC2Ev() {
define weak_odr void @_ZN3fooIiEC2Ev() comdat($_ZN3fooIiEC5Ev) {
  ret void
}
