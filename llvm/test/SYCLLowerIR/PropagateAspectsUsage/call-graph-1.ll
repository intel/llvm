; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s
;
; Test checks that the pass is able to propagate information about used aspects
; through a call graph
;
;   K1  K2
;  /  \/  \
; F1  F2   F3
;
; F1 doesn't use optional type.
; F2 uses optional A.
; F3 uses optional B.

%Optional.A = type { i32 }
%Optional.B = type { i32 }

; CHECK: spir_kernel void @kernel1() !sycl_used_aspects ![[#ID1:]]
define spir_kernel void @kernel1() {
  call spir_func void @func1()
  call spir_func void @func2()
  ret void
}

; CHECK: spir_kernel void @kernel2() !sycl_used_aspects ![[#ID2:]]
define spir_kernel void @kernel2() {
  call spir_func void @func2()
  call spir_func void @func3()
  ret void
}

; CHECK: spir_func void @func1() {
define spir_func void @func1() {
  %tmp = alloca i32
  ret void
}

; CHECK: spir_func void @func2() !sycl_used_aspects ![[#ID1]] {
define spir_func void @func2() {
  %tmp = alloca %Optional.A
  ret void
}

; CHECK: spir_func void @func3() !sycl_used_aspects ![[#ID3:]] {
define spir_func void @func3() {
  %tmp = alloca %Optional.B
  ret void
}

!sycl_types_that_use_aspects = !{!0, !1}
!0 = !{!"Optional.A", i32 1}
!1 = !{!"Optional.B", i32 2}

!sycl_aspects = !{!2}
!2 = !{!"fp64", i32 6}

; CHECK: ![[#ID1]] = !{i32 1}
; CHECK: ![[#ID2]] = !{i32 1, i32 2}
; CHECK: ![[#ID3]] = !{i32 2}
