; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-constructor-kind=none -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK: @__AsanKernelMetadata = appending dso_local local_unnamed_addr addrspace(1) global
; CHECK: @__AsanLaunchInfo = external addrspace(3) global ptr addrspace(1)

define spir_kernel void @sycl_kernel1() #0 {
; CHECK-LABEL: define spir_kernel void @sycl_kernel1(ptr addrspace(1) noundef %__asan_launch)
entry:
  ; store ptr addrspace(1) %__asan_launch, ptr addrspace(3) @__AsanLaunchInfo, align 8
  ret void
}

define spir_kernel void @sycl_kernel2() #0 {
; CHECK-LABEL: define spir_kernel void @sycl_kernel2(ptr addrspace(1) noundef %__asan_launch)
entry:
  ; CHECK: store ptr addrspace(1) %__asan_launch, ptr addrspace(3) @__AsanLaunchInfo, align 8
  call void @sycl_kernel1()
  ; CHECK: call void @sycl_kernel1(ptr addrspace(1) %__asan_launch)
  ret void
}

attributes #0 = { sanitize_address }
;; sycl-device-global-size = 16 * 2
;; sycl-host-access = 0 read-only
; CHECK: attributes #{{.*}} = { "sycl-device-global-size"="32" "sycl-device-image-scope" "sycl-host-access"="0" "sycl-unique-id"="_Z20__AsanKernelMetadata" }
