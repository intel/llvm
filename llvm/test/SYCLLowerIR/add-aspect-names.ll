; RUN: opt -passes=record-sycl-aspect-names -S < %s | FileCheck %s
;
; Basic add-aspect-names functionality test. Checks that 
; the !sycl_used_apsects metadata is updated from just being
; integer values to aspect value/name pairs as determined
; by !sycl_aspects.

; e.g. !sycl_used_aspects = !{i32 1, i32 0}
; =>   !sycl_used_aspects = !{!{!"B", i32 1}, !{!"A", i32 0}}

; Additionally checks that when there is no association 
; for a given aspect value, that metadata remains unchanged. 

; e.g. !sycl_used_aspects = !{i32 4, i32 0}
; =>   !sycl_used_aspects = !{i32 4, !{!"A", i32 0}} 

%A = type { i32 }
%B = type { i32 }
%C = type { i32 }
%D = type { i32 }

; CHECK: funcA() !sycl_used_aspects ![[fA:[0-9]+]]
define spir_func void @funcA() !sycl_used_aspects !5 {
  %tmp = alloca %A, align 8
  ret void
}

; CHECK: funcB() !sycl_used_aspects ![[fB:[0-9]+]]
define spir_func void @funcB() !sycl_used_aspects !6 {
  %tmp = alloca %B, align 8
  call spir_func void @funcA()
  ret void
}

; CHECK: funcC() !sycl_used_aspects ![[fC:[0-9]+]]
define spir_func void @funcC() !sycl_used_aspects !7 {
  %tmp = alloca %C, align 8
  call spir_func void @funcB()
  ret void
}

; CHECK: funcD() !sycl_used_aspects ![[fD:[0-9]+]]
define spir_func void @funcD() !sycl_used_aspects !8 {
  %tmp = alloca %D, align 8
  call spir_func void @funcC()
  ret void
}

define spir_kernel void @kernel() !sycl_used_aspects !8 !sycl_fixed_targets !9 {
  call spir_func void @funcD()
  ret void
}

!sycl_types_that_use_aspects = !{!0, !1, !2, !3}
!sycl_aspects = !{!0, !1, !2, !3, !4}

; CHECK-DAG: ![[mA:[0-9]+]] = !{!"A", i32 0}
; CHECK-DAG: ![[mB:[0-9]+]] = !{!"B", i32 1}
; CHECK-DAG: ![[mC:[0-9]+]] = !{!"C", i32 2}
; CHECK-DAG: ![[mD:[0-9]+]] = !{!"D", i32 3}
; CHECK-DAG: ![[fA]] = !{![[mA]]}
; CHECK-DAG: ![[fB]] = !{![[mB]], ![[mA]]}
; CHECK-DAG: ![[fC]] = !{![[mC]], ![[mB]], ![[mA]]}
; CHECK-DAG: ![[fD]] = !{![[mA]], ![[mB]], ![[mC]], ![[mD]], i32 4}

!0 = !{!"A", i32 0}
!1 = !{!"B", i32 1}
!2 = !{!"C", i32 2}
!3 = !{!"D", i32 3}
!4 = !{!"fp64", i32 6}
!5 = !{i32 0}
!6 = !{i32 1, i32 0}
!7 = !{i32 2, i32 1, i32 0}
!8 = !{i32 0, i32 1, i32 2, i32 3, i32 4}
!9 = !{}
