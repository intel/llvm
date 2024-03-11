; RUN: opt -passes=cleanup-sycl-metadata -S < %s | FileCheck %s
;
; Test checks that the pass is able to cleanup sycl_aspects and
; sycl_types_that_use_aspects module metadata

; ModuleID = 'multiple-aspects.ll'
source_filename = "multiple-aspects.ll"

%A = type { i32 }
%B = type { i32 }
%C = type { i32 }
%D = type { i32 }

define spir_func void @funcA() !sycl_used_aspects !5 {
  %tmp = alloca %A, align 8
  ret void
}

define spir_func void @funcB() !sycl_used_aspects !6 {
  %tmp = alloca %B, align 8
  call spir_func void @funcA()
  ret void
}

define spir_func void @funcC() !sycl_used_aspects !7 {
  %tmp = alloca %C, align 8
  call spir_func void @funcB()
  ret void
}

define spir_func void @funcD() !sycl_used_aspects !8 {
  %tmp = alloca %D, align 8
  call spir_func void @funcC()
  ret void
}

define spir_kernel void @kernel() !sycl_used_aspects !8 !sycl_fixed_targets !9 {
  call spir_func void @funcD()
  ret void
}

; CHECK-NOT: sycl_types_that_use_aspects
; CHECK-NOT: sycl_aspects
!sycl_types_that_use_aspects = !{!0, !1, !2, !3}
!sycl_aspects = !{!4}

!0 = !{!"A", i32 0}
!1 = !{!"B", i32 1}
!2 = !{!"C", i32 2}
!3 = !{!"D", i32 3, i32 4}
!4 = !{!"fp64", i32 6}
!5 = !{i32 0}
!6 = !{i32 1, i32 0}
!7 = !{i32 2, i32 1, i32 0}
!8 = !{i32 0, i32 1, i32 2, i32 3, i32 4}
!9 = !{}
