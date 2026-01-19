; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -S | FileCheck %s

; Check "sycl_service_kernel" function isn't instrumented.

target triple = "spir64-unknown-unknown"

%structtype = type { [3 x ptr addrspace(4)] }
%class.Base = type <{ ptr addrspace(4), i32, [4 x i8] }>

define linkonce_odr spir_func i32 @_ZTSN4sycl3_V16detail23__sycl_service_kernel__16AssertInfoCopierE(ptr addrspace(4) align 8 %this) sanitize_address "referenced-indirectly" {
; CHECK: @_ZTSN4sycl3_V16detail23__sycl_service_kernel__16AssertInfoCopierE{{.*}}#1
entry:
; CHECK-NOT: call void @__asan_load
  %base_data = getelementptr inbounds %class.Base, ptr addrspace(4) %this, i64 0, i32 1
  %1 = load i32, ptr addrspace(4) %base_data, align 8
  ret i32 %1
}

; CHECK: #1 {{.*}} disable_sanitizer_instrumentation
