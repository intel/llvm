; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s
;
; Test checks that the pass is able to propagate information about used aspects
; from getelementptr accesses to types that uses aspects.

%struct.StructWithAspect = type { i32 }

; CHECK: spir_kernel void @Kernel() !sycl_used_aspects ![[MDID:[0-9]+]]
define spir_kernel void @Kernel() {
entry:
  call spir_func void @ImmFunc()
  ret void
}

; CHECK: spir_func void @ImmFunc() !sycl_used_aspects ![[MDID]]
define spir_func void @ImmFunc() {
entry:
  %s = alloca %struct.StructWithAspect, align 4
  %s.ascast = addrspacecast ptr %s to ptr addrspace(4)
  call spir_func void @StructWithAspectCtor(ptr addrspace(4) noundef align 4 dereferenceable_or_null(4) %s.ascast)
  ret void
}

; CHECK: spir_func void @StructWithAspectCtor(ptr addrspace(4) noundef align 4 dereferenceable_or_null(4) %this) !sycl_used_aspects ![[MDID]]
define spir_func void @StructWithAspectCtor(ptr addrspace(4) noundef align 4 dereferenceable_or_null(4) %this) {
entry:
  %this.addr = alloca ptr addrspace(4), align 8
  %this.addr.ascast = addrspacecast ptr %this.addr to ptr addrspace(4)
  store ptr addrspace(4) %this, ptr addrspace(4) %this.addr.ascast, align 8
  %this1 = load ptr addrspace(4), ptr addrspace(4) %this.addr.ascast, align 8
  %a = getelementptr inbounds %struct.StructWithAspect, ptr addrspace(4) %this1, i32 0, i32 0
  store i32 0, ptr addrspace(4) %a, align 4
  ret void
}

!sycl_types_that_use_aspects = !{!1}
!sycl_aspects = !{!2}

!1 = !{!"struct.StructWithAspect", i32 5}
!2 = !{!"fp64", i32 6}

; CHECK: ![[MDID]] = !{i32 5}
