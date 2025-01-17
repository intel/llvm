; RUN: sycl-post-link -split=auto -properties -S < %s -o %t.table
; RUN: FileCheck %s --input-file=%t.table --check-prefix=CHECK-TABLE
; RUN: FileCheck %s --input-file=%t_0.ll --check-prefix=CHECK-FP64-SPLIT
; RUN: FileCheck %s --input-file=%t_1.ll --check-prefix=CHECK-FP64-DUMMY
; RUN: FileCheck %s --input-file=%t_1.prop --check-prefix=CHECK-FP64-DUMMY-PROPS
; RUN: FileCheck %s --input-file=%t_2.ll --check-prefix=CHECK-FP32-SPLIT

; CHECK-TABLE:      _0.prop
; CHECK-TABLE-NEXT: _1.prop
; CHECK-TABLE-NEXT: _2.prop

; CHECK-FP64-SPLIT: define spir_func void @bar()
; CHECK-FP32-SPLIT: define spir_func void @foo()

; CHECK-FP64-DUMMY: define spir_func void @bar()
; CHECK-FP64-DUMMY-NEXT: entry:
; CHECK-FP64-DUMMY-NEXT: ret void

; CHECK-FP64-DUMMY-PROPS: dummy-image=1

define spir_func void @foo() #1 {
  %x = alloca float
  ret void
}

define spir_func void @bar() #1 !sycl_used_aspects !1 {
  %x = alloca double
  %d = load double, ptr %x
  %res = fadd double %d, %d
  ret void
}

attributes #1 = { "sycl-module-id"="v.cpp" "indirectly-callable"="setA" }

!sycl_aspects = !{!0}
!0 = !{!"fp64", i32 6}
!1 = !{i32 6}
