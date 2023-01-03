; RUN: opt -passes=sycl-propagate-aspects-usage %s -S | FileCheck %s

;   K
;  /  \
; F1  F2
;  \  / \
;   F3   F4

; CHECK: spir_kernel void @kernel() !sycl_used_aspects ![[#ID1:]]
define spir_kernel void @kernel() {
  call spir_func void @func1()
  call spir_func void @func2()
  ret void
}

; CHECK: spir_func void @func1() !sycl_used_aspects ![[#ID2:]] {
define spir_func void @func1() {
  call spir_func void @func3()
  ret void
}

; CHECK: spir_func void @func2() !sycl_used_aspects ![[#ID1]] {
define spir_func void @func2() {
  call spir_func void @func3()
  call spir_func void @func4()
  ret void
}

; CHECK: spir_func void @func3() !sycl_used_aspects ![[#ID2]] {
define spir_func void @func3() !sycl_used_aspects !4 {
  ret void
}

; CHECK: spir_func void @func4() !sycl_used_aspects ![[#ID3:]]
; CHECK-SAME: !sycl_declared_aspects ![[#ID3]] {
define spir_func void @func4() !sycl_declared_aspects !3 {
  ret void
}

!sycl_aspects = !{!0, !1, !2}

!0 = !{!"host", i32 0}
!1 = !{!"cpu", i32 1}
!2 = !{!"fp64", i32 6}
!3 = !{i32 0}
!4 = !{i32 1}
!5 = !{i32 0, i32 1}

; CHECK: ![[#ID1]] = !{i32 1, i32 0}
; CHECK: ![[#ID2]] = !{i32 1}
; CHECK: ![[#ID3]] = !{i32 0}
