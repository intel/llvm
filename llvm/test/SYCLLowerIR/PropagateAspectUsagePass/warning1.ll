; RUN: opt --PropagateAspectUsage < %s > %t.ll 2>&1
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-FUNC
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-KERNEL
;
; Test checks warnings in case when intel_declared_aspects metadata
; doesn't cover all used aspects

%MyStruct1 = type { i32 }
%MyStruct2 = type { i32 }

; CHECK-FUNC-DAG: warning: function 'func' uses aspect '2' not listed in 'sycl::requires()'
; CHECK-FUNC-DAG: warning: function 'func' uses aspect '3' not listed in 'sycl::requires()'
define dso_local spir_func void @func() !intel_declared_aspects !0 {
  %tmp1 = alloca %MyStruct1
  %tmp2 = alloca %MyStruct2
  ret void
}

; CHECK-KERNEL: warning: function 'kernel' uses aspect '3' not listed in 'sycl::requires()'
; CHECK-KERNEL-NEXT: use is from this call chain:
; CHECK-KERNEL-NEXT: kernel()
; CHECK-KERNEL-NEXT: func()
; CHECK-KERNEL-NEXT: compile with '-g' to get source location
define dso_local spir_kernel void @kernel() !intel_declared_aspects !1 {
  call spir_func void @func()
  ret void
}


!0 = !{i32 1}
!1 = !{i32 2}

!intel_types_that_use_aspects = !{!2, !3}
!2 = !{!"MyStruct1", i32 2}
!3 = !{!"MyStruct2", i32 3}
