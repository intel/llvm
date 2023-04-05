; RUN: opt -passes=sycl-propagate-aspects-usage -sycl-propagate-aspects-usage-exclude-aspects=aspect4,aspect1 -S < %s | FileCheck %s
;
; Test checks that the pass is able to collect all aspects used in a function

%A = type { i32 }
%B = type { i32 }
%C = type { i32 }
%D = type { i32 }

; None of funcA's aspects are excluded.
; CHECK: define spir_func void @funcA() !sycl_used_aspects ![[#ID0:]] {
define spir_func void @funcA() {
  %tmp = alloca %A
  ret void
}

; funcB uses "aspect1" which is excluded, so the resulting aspects are the same
; as for funcA.
; CHECK: define spir_func void @funcB() !sycl_used_aspects ![[#ID0]] {
define spir_func void @funcB() {
  %tmp = alloca %B
  call spir_func void @funcA()
  ret void
}

; funcC has an aspect excluded, propagated from funcB.
; CHECK: define spir_func void @funcC() !sycl_used_aspects ![[#ID1:]] {
define spir_func void @funcC() {
  %tmp = alloca %C
  call spir_func void @funcB()
  ret void
}

; funcD has two aspects excluded; one from the use of D and one from propagated.
; from funcB and funcC.
; CHECK: define spir_func void @funcD() !sycl_used_aspects ![[#ID2:]] {
define spir_func void @funcD() {
  %tmp = alloca %D
  call spir_func void @funcC()
  ret void
}

; kernel1 has the same aspects as funcD.
; CHECK: define spir_kernel void @kernel1() !sycl_used_aspects ![[#ID2]]
define spir_kernel void @kernel1() {
  call spir_func void @funcD()
  ret void
}

; funcE should get none of its explicitly declared aspects in its
; sycl_used_aspects
; CHECK: define spir_func void @funcE() !sycl_declared_aspects ![[#DA1:]] {
define spir_func void @funcE() !sycl_declared_aspects !10 {
  ret void
}

; funcF should have the same aspects as funcE
; CHECK-NOT: define spir_func void @funcF() {{.*}} !sycl_used_aspects
define spir_func void @funcF() {
  call spir_func void @funcE()
  ret void
}

; funcG only keeps one aspect, the rest are excluded
; CHECK: define spir_func void @funcG() !sycl_declared_aspects ![[#DA2:]] !sycl_used_aspects ![[#ID3:]]
define spir_func void @funcG() !sycl_declared_aspects !11 {
  ret void
}

; funcH should have the same aspects as funcG
; CHECK: define spir_func void @funcH() !sycl_used_aspects ![[#ID3]]
define spir_func void @funcH() {
  call spir_func void @funcG()
  ret void
}

; CHECK: define spir_kernel void @kernel2() !sycl_used_aspects ![[#ID3]]
define spir_kernel void @kernel2() {
  call spir_func void @funcF()
  call spir_func void @funcH()
  ret void
}

; CHECK: define spir_func void @funcI() !sycl_used_aspects ![[#DA1]] {
define spir_func void @funcI() !sycl_used_aspects !10 {
  ret void
}

; CHECK-NOT: define spir_func void @funcJ() {{.*}} !sycl_used_aspects
define spir_func void @funcJ() {
  call spir_func void @funcI()
  ret void
}

;
; Note that the listed aspects can be reordered due to the merging of the
; aspect sets.
; CHECK: define spir_func void @funcK() !sycl_used_aspects ![[#ID4:]] {
define spir_func void @funcK() !sycl_used_aspects !11 {
  ret void
}

; CHECK: define spir_func void @funcL() !sycl_used_aspects ![[#ID3]]
define spir_func void @funcL() {
  call spir_func void @funcK()
  ret void
}

; CHECK: define spir_kernel void @kernel3() !sycl_used_aspects ![[#ID3]]
define spir_kernel void @kernel3() {
  call spir_func void @funcK()
  call spir_func void @funcL()
  ret void
}

!sycl_types_that_use_aspects = !{!0, !1, !2, !3}
!0 = !{!"A", i32 0}
!1 = !{!"B", i32 1}
!2 = !{!"C", i32 2}
!3 = !{!"D", i32 3, i32 4}

!sycl_aspects = !{!4, !5, !6, !7, !8, !9}
!4 = !{!"aspect0", i32 0}
!5 = !{!"aspect1", i32 1}
!6 = !{!"aspect2", i32 2}
!7 = !{!"aspect3", i32 3}
!8 = !{!"aspect4", i32 4}
!9 = !{!"fp64", i32 5}

!10 = !{i32 1}
!11 = !{i32 4, i32 2, i32 1}
; CHECK-DAG: ![[#DA1]] = !{i32 1}
; CHECK-DAG: ![[#DA2]] = !{i32 4, i32 2, i32 1}

; CHECK-DAG: ![[#ID0]] = !{i32 0}
; CHECK-DAG: ![[#ID1]] = !{i32 2, i32 0}
; CHECK-DAG: ![[#ID2]] = !{i32 0, i32 2, i32 3}
; CHECK-DAG: ![[#ID3]] = !{i32 2}
; CHECK-DAG: ![[#ID4]] = !{i32 2, i32 4, i32 1}
