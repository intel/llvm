; RUN: opt -passes=sycl-propagate-aspects-usage -S < %s | FileCheck %s
;
; Test checks that the pass is able to collect all aspects used in a function

%A = type { i32 }
%B = type { i32 }
%C = type { i32 }
%D = type { i32 }

; CHECK: define spir_func void @funcA() !sycl_used_aspects ![[#ID0:]] {
define spir_func void @funcA() {
  %tmp = alloca %A
  ret void
}

; CHECK: define spir_func void @funcB() !sycl_used_aspects ![[#ID1:]] {
define spir_func void @funcB() {
  %tmp = alloca %B
  call spir_func void @funcA()
  ret void
}

; CHECK: define spir_func void @funcC() !sycl_used_aspects ![[#ID2:]] {
define spir_func void @funcC() {
  %tmp = alloca %C
  call spir_func void @funcB()
  ret void
}

; CHECK: define spir_func void @funcD() !sycl_used_aspects ![[#ID3:]] {
define spir_func void @funcD() {
  %tmp = alloca %D
  call spir_func void @funcC()
  ret void
}

; CHECK: define spir_kernel void @kernel() !sycl_used_aspects ![[#ID3]]
define spir_kernel void @kernel() {
  call spir_func void @funcD()
  ret void
}

!sycl_types_that_use_aspects = !{!0, !1, !2, !3}
!0 = !{!"A", i32 0}
!1 = !{!"B", i32 1}
!2 = !{!"C", i32 2}
!3 = !{!"D", i32 3, i32 4}

!sycl_aspects = !{!4}
!4 = !{!"fp64", i32 6}

; CHECK: ![[#ID0]] = !{i32 0}
; CHECK: ![[#ID1]] = !{i32 1, i32 0}
; CHECK: ![[#ID2]] = !{i32 2, i32 1, i32 0}
; CHECK: ![[#ID3]] = !{i32 0, i32 1, i32 2, i32 3, i32 4}
