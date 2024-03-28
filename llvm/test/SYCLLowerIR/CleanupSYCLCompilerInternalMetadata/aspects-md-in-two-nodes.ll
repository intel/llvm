; RUN: opt -passes=cleanup-sycl-metadata -S < %s | FileCheck %s
;
; Test checks that the pass is able to cleanup sycl_aspects and
; sycl_types_that_use_aspects module metadata

; ModuleID = 'funny-aspects.ll'
source_filename = "funny-aspects.ll"

%A = type { i32 }

define spir_kernel void @kernel() !artificial !0 {
  ret void
}

; CHECK-NOT: sycl_types_that_use_aspects
; CHECK: !0 = !{!"A", i32 0}

!sycl_types_that_use_aspects = !{!0}

!0 = !{!"A", i32 0}
!1 = !{!"fp64", i32 6}
