; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -tsan-instrument-func-entry-exit=0 -tsan-instrument-memintrinsics=0 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

; CHECK: @__TsanKernelMetadata
; CHECK-SAME: [0 x { i64, i64 }]

; Function Attrs: sanitize_thread
define spir_kernel void @test(ptr addrspace(4) %a) #0 {
; CHECK-LABEL: void @test
entry:
  %tmp1 = load i8, ptr addrspace(4) %a, align 1
  %inc = add i8 %tmp1, 1
  ; CHECK-NOT: call void @__tsan_write
  store i8 %inc, ptr addrspace(4) %a, align 1
  ret void
}

; Function Attrs: sanitize_thread
define spir_kernel void @test_esimd(ptr addrspace(4) %a) #0 !sycl_explicit_simd !0 {
; CHECK-LABEL: void @test_esimd
entry:
  %tmp1 = load i16, ptr addrspace(4) %a, align 2
  %inc = add i16 %tmp1, 1
  ; CHECK-NOT: call void @__tsan_write
  store i16 %inc, ptr addrspace(4) %a, align 2
  ret void
}

attributes #0 = { sanitize_thread }

!0 = !{}
