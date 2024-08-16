; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_kernel void @sycl_kernel(ptr addrspace(1) %p) #0 {
; CHECK-LABEL: define spir_kernel void @sycl_kernel(ptr addrspace(1) %p, ptr addrspace(1) %__asan_launch) #0
entry:
  %0 = load i32, ptr addrspace(1) %p, align 4
  ; CHECK: store ptr addrspace(1) %__asan_launch, ptr addrspace(3) @__AsanLaunchInfo, align 8
  ; CHECK: call void @__asan_load4
  ret void
}

define spir_kernel void @esimd_kernel(ptr addrspace(1) %p) #0 !sycl_explicit_simd !1 {
; CHECK-LABEL: define spir_kernel void @esimd_kernel(ptr addrspace(1) %p, ptr addrspace(1) %__asan_launch) #0
entry:
  %0 = load i32, ptr addrspace(1) %p, align 4
  ; CHECK-NOT: store ptr addrspace(1) %__asan_launch, ptr addrspace(3) @__AsanLaunchInfo, align 8
  ; CHECK-NOT: call void @__asan_load4
  ret void
}

attributes #0 = { sanitize_address }
!1 = !{}
