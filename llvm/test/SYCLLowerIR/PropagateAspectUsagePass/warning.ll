; RUN: opt --PropagateAspectUsage < %s 2>&1 | FileCheck %s
;
; Test checks warnings in case when intel_declared_aspects metadata
; doesn't cover all used aspects

%MyStruct1 = type { i32 }
%MyStruct2 = type { i32 }

; CHECK-DAG: warning: for function "kernel" there is the list of missed aspects: [2, 3]
define dso_local spir_kernel void @kernel() !intel_declared_aspects !0 {
  %tmp = alloca %MyStruct2
  call spir_func void @func()
  ret void
}

; CHECK-DAG: warning: for function "func" there is the list of missed aspects: [2]
define dso_local spir_func void @func() !intel_declared_aspects !0 {
  %tmp = alloca %MyStruct1
  ret void
}

!0 = !{i32 1}

!intel_types_that_use_aspects = !{!1, !2}
!1 = !{!"MyStruct1", i32 2}
!2 = !{!"MyStruct2", i32 2, i32 3}
