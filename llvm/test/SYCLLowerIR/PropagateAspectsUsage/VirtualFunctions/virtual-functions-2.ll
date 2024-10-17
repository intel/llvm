; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s

%Foo = type { i32 }
%Bar = type { i32 }

; CHECK: @vfnFoo() #0 !sycl_used_aspects ![[#aspectsFoo:]]
define spir_func void @vfnFoo() #0 {
  %tmp = alloca %Foo
  ret void
}

; CHECK: @vfnBar() #1 !sycl_used_aspects ![[#aspectsBar:]]
define spir_func void @vfnBar() #1 {
  %tmp = alloca %Bar
  ret void
}

; CHECK: @kernel() #2 !sycl_used_aspects ![[#aspectsKernel:]]
define spir_kernel void @kernel() #2 {
  ret void
}

; CHECK: ![[#aspectsFoo]] = !{i32 1}
; CHECK: ![[#aspectsBar]] = !{i32 2}
; CHECK: ![[#aspectsKernel]] = !{i32 1, i32 2}

attributes #0 = { "indirectly-callable"="setFoo" }
attributes #1 = { "indirectly-callable"="setBar" }
attributes #2 = { "calls-indirectly"="setFoo,setBar" }

!sycl_aspects = !{!0}
!0 = !{!"fp64", i32 6}

!sycl_types_that_use_aspects = !{!1, !2}
!1 = !{!"Foo", i32 1}
!2 = !{!"Bar", i32 2}
