; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s
;
; Test checks that the pass is able to propagate information about conditionally
; and unconditionally used aspects through a call graph
;
;       K
;    /     \
;  CF1     CF2
;   |       |
;  F1 (A)  F2 (B)
;

%Optional.A = type { i32 }
%Optional.B = type { i32 }

%class.anon = type { ptr addrspace(4) }

; CHECK: define spir_kernel void @kernel() !sycl_conditionally_used_aspects ![[#ID1:]]
define spir_kernel void @kernel() {
  %agg.tmp = alloca %class.anon, align 8
  call spir_func void @cond_func1(ptr @func1, ptr %agg.tmp, i32 0, i32 0)
  call spir_func void @cond_func2(ptr @func2, ptr %agg.tmp, i32 0, i32 0)
  ret void
}

; CHECK: define spir_func void @func1() !sycl_conditionally_used_aspects ![[#ID9:]]
define spir_func void @func1() {
  %tmp = alloca %Optional.A
  ret void
}

; CHECK: define spir_func void @func2() !sycl_conditionally_used_aspects ![[#ID10:]]
define spir_func void @func2() {
  %tmp = alloca %Optional.B
  ret void
}

; CHECK: declare !sycl_conditionally_used_aspects ![[#ID9:]] spir_func void @cond_func1
declare spir_func void @cond_func1(ptr noundef byval(%fn) align 8, ptr noundef, i32 noundef, i32) #1

; CHECK: declare !sycl_conditionally_used_aspects ![[#ID10:]] spir_func void @cond_func2
declare spir_func void @cond_func2(ptr noundef byval(%fn) align 8, ptr noundef, i32 noundef, i32) #1

%fn = type { ptr addrspace(4) }

attributes #1 = { "sycl-call-if-on-device-conditionally"="true" }

!sycl_types_that_use_aspects = !{!0, !1}
!0 = !{!"Optional.A", i32 1}
!1 = !{!"Optional.B", i32 2}

!sycl_aspects = !{!2}
!2 = !{!"fp64", i32 6}

; TODO: need to optimize Conditions-Aspects pairs to combine Aspects from different pairs which have the same Conditions

; CHECK: ![[#ID1]] = !{![[#ID2:]], ![[#ID5:]], ![[#ID7:]]}
; CHECK: ![[#ID2]] = !{![[#ID3:]], ![[#ID4:]]}
; CHECK: ![[#ID3]] = !{i32 0}
; CHECK: ![[#ID4]] = !{i32 1}
; CHECK: ![[#ID5]] = !{![[#ID3:]], ![[#ID6:]]}
; CHECK: ![[#ID6]] = !{i32 1, i32 2}
; CHECK: ![[#ID7]] = !{![[#ID3:]], ![[#ID8:]]}
; CHECK: ![[#ID8]] = !{i32 2}
; CHECK: ![[#ID9]] = !{![[#ID2:]]}
; CHECK: ![[#ID10]] = !{![[#ID7:]]}
