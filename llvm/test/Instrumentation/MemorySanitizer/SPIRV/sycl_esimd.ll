; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -msan-spir-privates=0 -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

;CHECK: @__MsanKernelMetadata
;CHECK-SAME: [0 x { i64, i64, i64 }]

define spir_kernel void @test(ptr addrspace(1) noundef align 4 %_arg_array) sanitize_memory {
; CHECK-LABEL: define spir_kernel void @test
entry:
  %0 = load i32, ptr addrspace(1) %_arg_array, align 4
  %call = call spir_func i32 @foo(i32 %0)
  store i32 %call, ptr addrspace(1) %_arg_array, align 4
; CHECK-NOT: call void @__msan_maybe_warning
  ret void
}

define spir_kernel void @test_esimd(ptr addrspace(1) noundef align 4 %_arg_array) sanitize_memory !sycl_explicit_simd !0 {
; CHECK-LABEL: define spir_kernel void @test_esimd
entry:
  %0 = load i32, ptr addrspace(1) %_arg_array, align 4
  %call = call spir_func i32 @foo(i32 %0)
  store i32 %call, ptr addrspace(1) %_arg_array, align 4
; CHECK-NOT: call void @__msan_maybe_warning
  ret void
}

define spir_func i32 @foo(i32 %data) sanitize_memory {
entry:
  ret i32 %data
}

!0 = !{}
