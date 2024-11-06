; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s
;
; Test checks that the pass is able to propagate information about conditionally
; and unconditionally used aspects through a call graph
;
;       K
;    /     \
;  F1 (A)   F2 (B)
;            |
;           CF1
;            |
;           F3 (C)
;            |
;           CF2
;            |
;           F4 (D)
;

%Optional.A = type { i32 }
%Optional.B = type { i32 }
%Optional.C = type { i32 }
%Optional.D = type { i32 }

%class.anon = type { ptr addrspace(4) }

; CHECK: define spir_kernel void @kernel() !sycl_used_aspects ![[#ID1:]] !sycl_conditionally_used_aspects ![[#ID2:]]
define spir_kernel void @kernel() {
  call spir_func void @func1()
  call spir_func void @func2()
  ret void
}

; CHECK: define spir_func void @func1() !sycl_used_aspects ![[#ID8:]]
define spir_func void @func1() {
  %tmp = alloca %Optional.A
  ret void
}

; CHECK: define spir_func void @func2() !sycl_used_aspects ![[#ID9:]] !sycl_conditionally_used_aspects ![[#ID2:]]
define spir_func void @func2() {
  %tmp = alloca %Optional.B
  %agg.tmp = alloca %class.anon, align 8
  call spir_func void @cond_func1(ptr @func3, ptr %agg.tmp, i32 1, i32 2)
  ret void
}

; CHECK: define spir_func void @func3() !sycl_conditionally_used_aspects ![[#ID2:]]
define spir_func void @func3() {
  %tmp = alloca %Optional.C
  %agg.tmp = alloca %class.anon, align 8
  call spir_func void @cond_func2(ptr @func4, ptr %agg.tmp, i32 3, i32 4)
  ret void
}

; CHECK: define spir_func void @func4() !sycl_conditionally_used_aspects ![[#ID10:]]
define spir_func void @func4() {
  %tmp = alloca %Optional.D
  ret void
}

; CHECK: declare !sycl_conditionally_used_aspects ![[#ID2:]] spir_func void @cond_func1
declare spir_func void @cond_func1(ptr noundef byval(%fn) align 8, ptr noundef, i32 noundef, i32) #1

; CHECK: declare !sycl_conditionally_used_aspects ![[#ID10:]] spir_func void @cond_func2
declare spir_func void @cond_func2(ptr noundef byval(%fn) align 8, ptr noundef, i32 noundef, i32) #1

%fn = type { ptr addrspace(4) }

attributes #1 = { "sycl-call-if-on-device-conditionally"="true" }

!sycl_types_that_use_aspects = !{!0, !1, !2, !3}
!0 = !{!"Optional.A", i32 1}
!1 = !{!"Optional.B", i32 2}
!2 = !{!"Optional.C", i32 3}
!3 = !{!"Optional.D", i32 4}

!sycl_aspects = !{!4}
!4 = !{!"fp64", i32 6}

; CHECK: ![[#ID1]] = !{i32 1, i32 2}
; CHECK: ![[#ID2]] = !{![[#ID3:]], ![[#ID5:]]}
; CHECK: ![[#ID3]] = !{![[#ID1:]], ![[#ID4:]]}
; CHECK: ![[#ID4]] = !{i32 3, i32 4}
; CHECK: ![[#ID5]] = !{![[#ID6:]], ![[#ID7:]]}
; CHECK: ![[#ID6]] = !{i32 1, i32 2, i32 3, i32 4}
; CHECK: ![[#ID7]] = !{i32 4}
; CHECK: ![[#ID8]] = !{i32 1}
; CHECK: ![[#ID9]] = !{i32 2}
; CHECK: ![[#ID10]] = !{![[#ID5:]]}
