; This tests ensures that the aspects of a global variable are propagated.
; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s

%struct.StructWithAspect = type { i32 }
@global = internal addrspace(3) global %struct.StructWithAspect undef, align 8

declare void @external(ptr addrspace(4))

; CHECK: spir_kernel void @foo() !sycl_used_aspects ![[MDID:[0-9]+]]
define spir_kernel void @foo() {
  %res = load ptr addrspace(3), ptr addrspace(3) @global
  ret void
}

; CHECK: spir_kernel void @bar() !sycl_used_aspects ![[MDID]]
define spir_kernel void @bar() {
  call void @external(ptr addrspace(4) addrspacecast(ptr addrspace(3) @global to ptr addrspace(4)))
  ret void
}

!sycl_types_that_use_aspects = !{!1}
!sycl_aspects = !{!2}

!1 = !{!"struct.StructWithAspect", i32 6}
!2 = !{!"fp64", i32 6}

; CHECK: ![[MDID]] = !{i32 6}