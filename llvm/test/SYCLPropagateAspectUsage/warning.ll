; RUN opt --PropagateAspectUsage < %s >/dev/null 2>1 | FileCheck %s
;
; Test check warnings in case when intel_declared_aspects metadata
; doesn't cover all used aspects

%MyStruct1 = type { i32 }
%MyStruct2 = type { i32 }

; CHECK: warning: for function "kernel" aspects [2, 3] are missed in function declaration
define dso_local spir_kernel void @kernel() !intel_declared_aspects !0 {
  %tmp = alloca %MyStruct2
  call spir_func void @func()
  ret void
}

; CHECK: warning: for function "func" aspect 2 is missed in function declaration
define dso_local spir_func void @func() !intel_declared_aspects !0 {
  %tmp = alloca %MyStruct1
  ret void
}

!0 = !{i32 1}

!intel_types_that_use_aspects = !{!1, !2}
!1 = !{!"MyStruct1", i32 2}
!2 = !{!"MyStruct2", i32 2, i32 3}
