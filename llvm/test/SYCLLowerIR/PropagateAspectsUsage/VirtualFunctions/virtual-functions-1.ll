; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s

; CHECK: @vfn() #0 !sycl_used_aspects ![[#aspects:]]
define spir_func void @vfn() #0 {
  %tmp = alloca double
  ret void
}

; CHECK: @foo() #1 !sycl_used_aspects ![[#aspects]]
define spir_kernel void @foo() #1 {
  ret void
}

; CHECK: ![[#aspects]] = !{i32 6}

attributes #0 = { "indirectly-callable"="_ZTSv" }
attributes #1 = { "calls-indirectly"="_ZTSv" }

!sycl_aspects = !{!0}
!0 = !{!"fp64", i32 6}
