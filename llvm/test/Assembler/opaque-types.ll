; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; Check support for basic opaque type properties

@global = global opaque("ctype") zeroinitializer

define opaque("atype") @foo(opaque("atype") %a) {
  ret opaque("atype") %a
}

define opaque("btype") @func2() {
  %mem = alloca opaque("btype")
  %val = load opaque("btype"), ptr %mem
  ret opaque("btype") poison
}

; CHECK: @global = global opaque("ctype") zeroinitializer
; CHECK: define opaque("atype") @foo(opaque("atype") %a) {
; CHECK:   ret opaque("atype") %a
; CHECK: }
; CHECK: define opaque("btype") @func2() {
; CHECK:   %mem = alloca opaque("btype")
; CHECK:   %val = load opaque("btype"), ptr %mem
; CHECK:   ret opaque("btype") poison
; CHECK: }
