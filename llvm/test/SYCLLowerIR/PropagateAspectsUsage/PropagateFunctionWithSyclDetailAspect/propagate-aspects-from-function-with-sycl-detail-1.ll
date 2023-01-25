; RUN: opt -passes=sycl-propagate-aspects-usage %s -S | FileCheck %s

; Check that baz() & bar() take the same !sycl_used_aspects as foo()

; baz()
;  |
;  v
; bar()
;  |
;  v
; foo()

source_filename = "main.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK: void @kernel() !sycl_used_aspects ![[#ASPECT:]]
define weak_odr dso_local spir_kernel void @kernel() {
entry:
  call spir_func void @_Z3bazv()
  ret void
}

; CHECK: void @_Z3bazv() !sycl_used_aspects ![[#ASPECT]] {
define dso_local spir_func void @_Z3bazv() {
entry:
  call spir_func void @_Z3barv()
  ret void
}

; CHECK: void @_Z3barv() !sycl_used_aspects ![[#ASPECT]] {
define dso_local spir_func void @_Z3barv() {
entry:
  call spir_func void @_Z3foov()
  ret void
}

; CHECK: void @_Z3foov() !sycl_used_aspects ![[#ASPECT]] {
define dso_local spir_func void @_Z3foov() !sycl_used_aspects !2 {
entry:
  ret void
}

!sycl_aspects = !{!0, !1}

!0 = !{!"gpu", i32 2}
!1 = !{!"fp64", i32 6}
!2 = !{i32 2}
