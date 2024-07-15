; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 | FileCheck %s

; Check referenced-indirectly function isn't instrumented.

target triple = "spir64-unknown-unknown"

%structtype = type { [3 x ptr addrspace(4)] }
%class.Base = type <{ ptr addrspace(4), i32, [4 x i8] }>
@_ZTV8Derived1 = linkonce_odr addrspace(1) constant %structtype { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @_ZN8Derived17displayEv to ptr addrspace(4))] }, align 8, !spirv.Decorations !0

define linkonce_odr spir_func i32 @_ZN8Derived17displayEv(ptr addrspace(4) align 8 %this) sanitize_address "referenced-indirectly" {
entry:
; CHECK-NOT: call void @__asan_load

  %base_data = getelementptr inbounds %class.Base, ptr addrspace(4) %this, i64 0, i32 1
  %1 = load i32, ptr addrspace(4) %base_data, align 8
  ret i32 %1
}

!0 = !{!1, !2, !3}
!1 = !{i32 22}
!2 = !{i32 41, !"_ZTV8Derived1", i32 2}
!3 = !{i32 44, i32 8}
