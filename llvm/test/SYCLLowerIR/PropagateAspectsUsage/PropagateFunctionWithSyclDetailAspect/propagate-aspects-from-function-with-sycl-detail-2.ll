; RUN: opt -passes=sycl-propagate-aspects-usage %s -S | FileCheck %s

; Check that baz() takes a mix of "!sycl_used_aspects" of bar() & foo()

;    baz()
;   /     \
;  v       v
; bar()   foo()

source_filename = "main.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK: void @_Z3bazv() !sycl_used_aspects ![[#ASPECT1:]]
define dso_local spir_kernel void @_Z3bazv() {
entry:
  call spir_func void @_Z3barv()
  call spir_func void @_Z3foov()
  ret void
}

; CHECK: void @_Z3barv() !sycl_used_aspects ![[#ASPECT2:]] {
define dso_local spir_func void @_Z3barv() !sycl_used_aspects !3 {
entry:
  ret void
}

; CHECK: void @_Z3foov() !sycl_used_aspects ![[#ASPECT3:]] {
define dso_local spir_func void @_Z3foov() !sycl_used_aspects !4 {
entry:
  ret void
}

; CHECK: ![[#ASPECT1]] = !{i32 2, i32 1}
; CHECK: ![[#ASPECT2]] = !{i32 2}
; CHECK: ![[#ASPECT3]] = !{i32 1}

!sycl_aspects = !{!0, !1, !2}

!0 = !{!"cpu", i32 1}
!1 = !{!"gpu", i32 2}
!2 = !{!"fp64", i32 6}
!3 = !{i32 2}
!4 = !{i32 1}
